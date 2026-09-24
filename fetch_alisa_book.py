#!/usr/bin/env python3
"""抓取 Notion 公开页面《Alisa's book of LLMs》，转成 Markdown 留档（保留公式与图片）。

用法：
  python3 fetch_alisa_book.py                     # 抓取并写出 alisa-book-of-llms.md
  python3 fetch_alisa_book.py --blocks b.json     # 用已缓存的 block 数据重新渲染
  python3 fetch_alisa_book.py --skip-images       # 只出正文，不下载图片

说明：
  - 网络请求统一走 curl：macOS 自带 curl 的证书链可用，而 python.org 版
    Python 3.14 常缺 CA 根证书，直接 urllib 抓 Notion 会 SSL 校验失败。
  - 图片需要两步：先请求 notion.site/image/... 拿 302，再下载 CDN 直链。
  - 公式全部保留 LaTeX 源码（行内 $...$，独立 $$...$$），不转图片。
"""

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

SITE = "https://alisawuffles.notion.site"
PAGE_URL = f"{SITE}/alisa-s-book-of-llms"
API = f"{SITE}/api/v3/loadCachedPageChunkV2"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

HEADINGS = {"header": "##", "sub_header": "###", "sub_sub_header": "####"}

LANG_ALIAS = {"plain text": "", "plaintext": "", "c++": "cpp", "shell": "bash"}

IMAGE_DIR = "images/alisa-book"


def curl(args, data=None, timeout=90):
    cmd = ["curl", "-sS", "-m", str(timeout), "-A", UA] + args
    if data is not None:
        cmd += ["--data-binary", "@-"]
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl 失败({p.returncode}): {p.stderr.decode(errors='replace')[:300]}")
    return p.stdout


def curl_retry(args, timeout=90, attempts=4):
    """CDN 偶发 Connection reset，重试几次。"""
    last = None
    for i in range(attempts):
        try:
            return curl(args, timeout=timeout)
        except RuntimeError as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def get_page_id(url=PAGE_URL):
    html = curl([url]).decode("utf-8", errors="replace")
    m = re.search(r'"pageId":"([0-9a-f-]{36})"', html)
    if not m:
        raise RuntimeError("页面 HTML 里没找到 pageId")
    return m.group(1)


def fetch_blocks(page_id, cache_path=None):
    blocks, chunk, cursor = {}, 0, {"stack": []}
    while True:
        body = json.dumps({
            "page": {"id": page_id}, "limit": 100, "cursor": cursor,
            "chunkNumber": chunk, "verticalColumns": False,
        }).encode()
        data = json.loads(curl(["-X", "POST", API, "-H", "Content-Type: application/json"], data=body))
        record_map = data.get("recordMap", {}).get("block", {}) or {}
        new = 0
        for bid, rec in record_map.items():
            if bid not in blocks:
                new += 1
            blocks[bid] = rec
        cursors = data.get("cursors") or []
        print(f"  第 {chunk + 1} 批：新 block {new}，累计 {len(blocks)}", file=sys.stderr)
        if new == 0 or not cursors:
            break
        cursor = {"stack": cursors[0]["stack"]}
        chunk += 1
        if chunk > 40:
            break
        time.sleep(0.3)
    if cache_path:
        Path(cache_path).write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    return blocks


def plain_text(rt):
    """把富文本数组拍平成纯文本（行内公式保留 LaTeX）。"""
    if not rt:
        return ""
    out = []
    for seg in rt:
        if not isinstance(seg, list) or not seg:
            continue
        text = seg[0]
        decs = seg[1] if len(seg) > 1 and isinstance(seg[1], list) else []
        for d in decs:
            if isinstance(d, list) and d and d[0] == "e" and len(d) > 1:
                text = f"${d[1]}$"
                break
        if text == "‣":
            continue
        out.append(text)
    return "".join(out)


def slugify(text, maxlen=60):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-").lower()
    text = re.sub(r"-{2,}", "-", text)
    return (text or "image")[:maxlen].strip("-") or "image"


class Book:
    def __init__(self, raw):
        self.blocks = {}
        for bid, rec in raw.items():
            value = rec.get("value")
            value = value.get("value") if isinstance(value, dict) else None
            if value:
                self.blocks[bid] = value
        self.images = []       # 按正文出现顺序
        self.image_index = {}  # block id -> 序号
        self.headers = []      # (level, text)
        self.section = []      # 当前所在章节标题，用于给无名图片起文件名
        self.section_slug = ""
        self.page_titles = {
            bid: plain_text(b.get("properties", {}).get("title"))
            for bid, b in self.blocks.items() if b.get("type") == "page"
        }

    # ---------- 行内富文本 ----------
    def rich(self, rt):
        if not rt:
            return ""
        return "".join(self.segment(s) for s in rt if isinstance(s, list) and s)

    def segment(self, seg):
        text = seg[0]
        decs = [d for d in (seg[1] if len(seg) > 1 and isinstance(seg[1], list) else [])
                if isinstance(d, list) and d]

        for d in decs:                                    # 行内公式
            if d[0] == "e" and len(d) > 1:
                return f"${d[1].strip()}$"

        if text == "‣":                                   # mention
            for d in decs:
                if d[0] == "p" and len(d) > 1:
                    title = self.page_titles.get(d[1], "链接页面")
                    return f"[{title}]({SITE}/{d[1].replace('-', '')})"
                if d[0] == "d" and len(d) > 1 and isinstance(d[1], dict):
                    start = (d[1].get("start_date") or "")[:10]
                    end = (d[1].get("end_date") or "")[:10]
                    return f"{start} – {end}" if end else start
            return ""

        href = next((d[1] for d in decs if d[0] == "a" and len(d) > 1), None)
        if any(d[0] == "c" for d in decs) and text.strip():
            out = f"`{text.strip()}`"
        else:
            out = text
        if out.strip() and any(d[0] == "b" for d in decs):
            out = f"**{out}**"
        if out.strip() and any(d[0] == "i" for d in decs):
            out = f"*{out}*"
        if out.strip() and any(d[0] == "s" for d in decs):
            out = f"~~{out}~~"
        if href:
            out = f"[{out}]({href})" if out.strip() else href
        return out

    # ---------- 块渲染 ----------
    LIST_TYPES = {"bulleted_list", "numbered_list", "to_do", "toggle"}

    def render(self, ids, depth=0):
        lines, number, prev_list = [], 0, False
        for bid in ids or []:
            block = self.blocks.get(bid)
            if not block:
                continue
            is_list = block["type"] in self.LIST_TYPES
            if prev_list and not is_list:
                lines.append("")          # 列表后面接标题/段落时必须空行，否则被吸进列表项
            number = number + 1 if block["type"] == "numbered_list" else 0
            lines.extend(self.render_block(block, depth, number))
            prev_list = is_list
        return lines

    def render_block(self, block, depth, number=0):
        t = block["type"]
        props = block.get("properties") or {}
        fmt = block.get("format") or {}
        kids = block.get("content") or []
        indent = "  " * depth
        title = self.rich(props.get("title"))
        lines = []

        if t == "page":
            return []

        if t in HEADINGS:
            level = len(HEADINGS[t])
            plain = title.replace("**", "").replace("~~", "").strip()
            lines.append(f"{HEADINGS[t]} {plain}".rstrip())
            self.headers.append((level, plain))
            self.section = self.section[: level - 2] + [plain]
            self.section_slug = slugify(" ".join(self.section[-2:]) or "figure")
            if kids:
                lines.append("")
                lines.extend(self.render(kids, depth + 1))

        elif t == "text":
            lines.append(f"{indent}{title}".rstrip())
            lines.extend(self.render(kids, depth + 1))
            lines.append("")

        elif t == "bulleted_list":
            lines.append(f"{indent}- {title}".rstrip())
            lines.extend(self.render(kids, depth + 1))

        elif t == "numbered_list":
            lines.append(f"{indent}{number}. {title}".rstrip())
            lines.extend(self.render(kids, depth + 1))

        elif t == "to_do":
            checked = "x" if (props.get("checked") or [["No"]])[0][0] == "Yes" else " "
            lines.append(f"{indent}- [{checked}] {title}".rstrip())
            lines.extend(self.render(kids, depth + 1))

        elif t == "toggle":
            lines.append(f"{indent}- {title}".rstrip())
            lines.extend(self.render(kids, depth + 1))

        elif t == "quote":
            lines.append(f"> {title}".rstrip())
            lines.extend(f"> {ln}" if ln else ">" for ln in self.render(kids, 0))
            lines.append("")

        elif t == "callout":
            icon = fmt.get("page_icon") or ""
            if isinstance(icon, str) and icon.startswith("/"):
                icon = ""
            lines.append(f"> {f'{icon} {title}'.strip()}".rstrip())
            lines.extend(f"> {ln}" if ln else ">" for ln in self.render(kids, 0))
            lines.append("")

        elif t == "code":
            lang = LANG_ALIAS.get(plain_text(props.get("language")).lower(),
                                  plain_text(props.get("language")).lower())
            lines.append(f"{indent}```{lang}")
            lines.extend(f"{indent}{ln}" for ln in plain_text(props.get("title")).split("\n"))
            lines.append(f"{indent}```")
            lines.append("")

        elif t == "equation":
            latex = plain_text(props.get("title"))
            lines.append(f"{indent}$$")
            lines.extend(f"{indent}{ln}" for ln in latex.split("\n"))
            lines += [f"{indent}$$", ""]

        elif t == "divider":
            lines += ["---", ""]

        elif t == "image":
            n = self.register_image(block)
            alt = self.images[n - 1]["alt"].replace("]", "\\]")
            lines += [f"{indent}![{alt}](@@IMG{n:02d}@@)", ""]

        elif t == "bookmark":
            link = plain_text(props.get("link")) or plain_text(props.get("title"))
            lines += [f"{indent}<{link}>", ""]

        elif t in ("video", "embed"):
            link = plain_text(props.get("source")) or plain_text(props.get("link"))
            lines += [f"{indent}[视频/嵌入]({link})", ""]

        elif t == "table":
            lines.extend(f"{indent}{ln}" for ln in self.render_table(block))
            lines.append("")

        elif t == "column_list":
            for cid in kids:
                col = self.blocks.get(cid)
                if col:
                    lines.extend(self.render(col.get("content") or [], depth))
            lines.append("")

        elif t == "column":
            lines.extend(self.render(kids, depth))

        elif t == "table_of_contents":
            lines += ["<!-- TOC -->", ""]

        elif t == "table_row":
            pass

        else:
            if title:
                lines += [f"{indent}{title}".rstrip(), ""]
            lines.extend(self.render(kids, depth + 1))

        return lines

    def register_image(self, block):
        props = block["properties"]
        caption = plain_text(props.get("caption")).strip()
        raw_title = plain_text(props.get("title")).strip()
        generic = raw_title.lower() in ("", "image", "image.png", "untitled", "untitled.png")
        alt = caption or (raw_title if not generic else "")
        if not alt:
            alt = f"figure {len(self.images) + 1}"
        hint = caption or (raw_title if not generic else "") or self.section_slug or "figure"
        if block["id"] not in self.image_index:
            self.image_index[block["id"]] = len(self.images) + 1
            self.images.append({
                "id": block["id"],
                "source": plain_text(props.get("source")),
                "alt": alt,
                "name_hint": hint,
                "space_id": block.get("space_id", ""),
            })
        return self.image_index[block["id"]]

    def render_table(self, block):
        rows = [self.blocks[r] for r in (block.get("content") or []) if r in self.blocks]
        if not rows:
            return []
        columns = list((rows[0].get("properties") or {}).keys())
        has_header = bool((block.get("format") or {}).get("table_block_column_header"))
        out = []
        for i, row in enumerate(rows):
            cells = [self.rich((row.get("properties") or {}).get(c)) for c in columns]
            cells = [c.replace("|", "\\|").replace("\n", "<br>").strip() for c in cells]
            out.append("| " + " | ".join(cells) + " |")
            if i == 0:
                out.append("|" + "|".join([" :-- "] * len(columns)) + "|" if has_header
                           else "|" + "|".join(["     "] * len(columns)) + "|")
        return out


def download_image(entry, index, images_dir):
    src = entry["source"]
    if src.startswith("http"):
        url = src
    else:
        url = (f"{SITE}/image/{urllib.parse.quote(src, safe='')}"
               f"?table=block&id={entry['id']}&spaceId={entry['space_id']}&userId=&cache=v2")
    headers = curl(["-D", "-", "-o", "/dev/null", url]).decode("utf-8", errors="replace")
    final = next((ln.split(":", 1)[1].strip() for ln in headers.splitlines()
                  if ln.lower().startswith("location:")), url)

    m = re.search(r"\.(png|jpe?g|gif|webp|svg|avif)$", src, re.I)
    ext = "." + m.group(1).lower() if m else ".png"
    name = re.sub(r"\.(png|jpe?g|gif|webp|svg|avif)$", "", entry.get("name_hint", ""), flags=re.I)
    stem = f"{index:02d}-{slugify(name)}"
    dest = Path(images_dir) / f"{stem}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(curl_retry(["-L", final], timeout=300))
    print(f"  图片 {index:02d}: {name or '(无题)'} -> {dest} ({dest.stat().st_size // 1024} KiB)",
          file=sys.stderr)
    return f"{images_dir.rstrip('/')}/{stem}{ext}"


TEMPLATE = """# Alisa's book of LLMs（本地留档）

> **原文**：<https://alisawuffles.notion.site/alisa-s-book-of-llms> —— Notion 公开页面，作者 Alisa Liu
> **抓取时间**：{date}
> **抓取方式**：Notion 公开页 `loadCachedPageChunkV2` 接口 → Markdown（脚本 `fetch_alisa_book.py`）
> **规模**：{n_blocks} 个 block、{n_eq} 个独立公式块、{n_img} 张图片（已下载到 `images/alisa-book/`）
> **版权**：第三方教材原文，仅作个人学习留档：未翻译、未核验，请勿转载或以任何形式公开分发。

## 待学习 TODO

- [ ] 通读一遍，标出与当前 infra 主线（MoE、并行策略、推理引擎、RLHF）直接相关的章节
- [ ] 精读 Transformer 与并行/分布式两章，把 {n_eq} 个公式逐个推一遍，对照本仓库已有笔记
- [ ] 核对 {n_img} 张图片与正文位置一一对应，公式在本地预览里渲染正常（Hugo 用 MathJax）
- [ ] 把有复用价值的章节重写成自己的中文笔记（保留推导与出处），不做原文搬运
- [ ] 与已有笔记互链：DeepSeek Infra Stack、vLLM 阅读导引、MoE / 算子主题

---

<!-- TOC -->

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", help="已缓存的 blocks.json（跳过联网抓取）")
    ap.add_argument("--out", default="alisa-book-of-llms.md")
    ap.add_argument("--images-dir", default=IMAGE_DIR)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    raw = (json.loads(Path(args.blocks).read_text(encoding="utf-8")) if args.blocks
           else fetch_blocks(get_page_id(), cache_path="/tmp/alisa-book-blocks.json"))

    book = Book(raw)
    root = next((b for b in book.blocks.values() if b["type"] == "page"), None)
    if not root:
        raise SystemExit("没找到根 page block")
    body = book.render(root.get("content") or [])

    toc = []
    for level, text in book.headers:
        anchor = "#" + re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.strip().lower()).strip("-")
        toc.append(f"{'  ' * (level - 2)}- [{text}]({anchor})")

    n_eq = sum(1 for b in book.blocks.values() if b["type"] == "equation")
    md = TEMPLATE.format(date=time.strftime("%Y-%m-%d"), n_blocks=len(book.blocks),
                         n_eq=n_eq, n_img=len(book.images))
    text = "\n".join(body).rstrip()
    text = text.replace("<!-- TOC -->", "\n".join(toc))
    md = md + text + "\n"

    for i, entry in enumerate(book.images, 1):
        if args.skip_images:
            rel = f"{args.images_dir.rstrip('/')}/{i:02d}-{slugify(entry['name_hint'])}.png"
        else:
            rel = download_image(entry, i, args.images_dir)
        md = md.replace(f"@@IMG{i:02d}@@", rel)

    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\n写出 {args.out}：{len(md)} 字符 / {md.count(chr(10)) + 1} 行，"
          f"公式块 {n_eq}，图片 {len(book.images)} 张", file=sys.stderr)


if __name__ == "__main__":
    main()
