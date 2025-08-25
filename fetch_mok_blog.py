#!/usr/bin/env python3
"""
抓取 Cursor 博客《Mixture-of-Kittens》，提取正文+图片，转 Markdown 存本地。

用法：
  python3 fetch_mok_blog.py <html_file> <output_dir>
"""
import os
import re
import sys
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup


def decode_next_image_url(src: str) -> str:
    """把 Next.js 图片优化 URL 解码成真实图片 URL。"""
    if not src.startswith("/marketing-static/_next/image"):
        return src
    parsed = urllib.parse.urlparse(src)
    query = urllib.parse.parse_qs(parsed.query)
    real_url = query.get("url", [""])[0]
    return urllib.parse.unquote(real_url) if real_url else src


def download_image(url: str, dest_dir: Path, index: int, alt_text: str) -> str:
    """下载图片到 dest_dir，返回本地相对路径。失败时返回空字符串。"""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    ext = ""
    if url.startswith("data:"):
        return ""  # data URI 跳过
    # 从 URL 推断扩展名
    path_part = urllib.parse.urlparse(url).path
    m = re.search(r"\.(png|jpe?g|webp|gif|svg|avif)$", path_part, re.I)
    if m:
        ext = "." + m.group(1).lower()
    if not ext:
        ext = ".png"  # 默认
    # 生成文件名：alt 文本（清理）+ 序号
    safe_alt = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", alt_text or "").strip("-")[:40]
    fname = f"mok-{index:02d}{'-' + safe_alt if safe_alt else ''}{ext}"
    dest = dest_dir / fname
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100:  # 空图/占位图
            return ""
        dest.write_bytes(data)
        return fname
    except Exception as exc:
        print(f"  ⚠️ 下载失败 {url[:100]}... ({exc})", file=sys.stderr)
        return ""


def main() -> int:
    html_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")

    # 1. 定位正文 article（包含 h1 标题的那个）
    articles = soup.find_all("article")
    article = None
    for a in articles:
        if a.find("h1") and "Mixture" in a.find("h1").get_text():
            article = a
            break
    if article is None:
        print("❌ 未找到正文 article", file=sys.stderr)
        return 1
    print(f"✅ 找到正文 article（长度 {len(article.get_text())}）")

    # 2. 提取元信息（日期、作者）
    meta_text = article.get_text(" ", strip=True)
    date_m = re.search(r"(Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)\s+\d{1,2},\s+\d{4}", meta_text)
    print(f"📅 日期: {date_m.group(0) if date_m else '未知'}")
    author_names = [img.get("alt", "").strip() for img in article.find_all("img") if img.get("alt")]

    # 3. 删除非正文元素（相关文章、导航、脚本、TOC、图表导出副本等）
    for selector in ["script", "style", "nav", "footer", "header"]:
        for el in article.find_all(selector):
            el.decompose()
    # 删除侧栏推荐文章卡片（文章标题不是本页标题的 article）
    for a in article.find_all("article"):
        h = a.find("h1") or a.find("h2") or a.find("h3")
        if h and "Mixture" not in h.get_text():
            a.decompose()
    # 删除目录 (TOC) 卡片：包含"目录"文本的 nav.card
    for nav_el in article.find_all("nav"):
        if "目录" in nav_el.get_text()[:20]:
            nav_el.decompose()
    # 删除图表导出副本容器（builder-export-chart 会有 2~3 份离屏渲染副本）
    for chart in article.find_all(class_=lambda c: c and "builder-export-chart" in c):
        chart.decompose()
    # 删除标题里的锚点链接垃圾（<a class="anchor-link">#</a>）
    for h in article.find_all(["h1", "h2", "h3", "h4"]):
        for anchor in h.find_all("a", class_="anchor-link"):
            anchor.decompose()
    # 删除装饰性 SVG 图标（链接/下载小图标，转换后会变成 base64 噪声）
    for svg in article.find_all("svg"):
        svg.decompose()
    # KaTeX 公式降级为纯文本（无 LaTeX 源可提取，span 文本即渲染内容）
    for katex in article.find_all(class_=lambda c: c and "katex" in c):
        if "katex-display" in (katex.get("class") or []):
            katex.replace_with("\n\n" + katex.get_text() + "\n\n")
        else:
            katex.replace_with(katex.get_text())
    # 清除空布局 div（无文本且无图片/表格/代码/公式的子节点）
    changed = True
    while changed:
        changed = False
        for div in article.find_all("div"):
            has_meaningful = div.find(["img", "table", "pre", "figure", "code"])
            if not has_meaningful and not div.get_text(strip=True):
                div.decompose()
                changed = True
    # 删除左侧面包屑导航栏（"Blog / 研究" 所在 sticky 列）
    for crumb_link in article.find_all("a", href="/cn/blog"):
        wrapper = crumb_link.find_parent(class_=lambda c: c and "col-span-full" in c and "xl:col-start-1" in c)
        if wrapper is not None:
            wrapper.decompose()
    # 展开 figure 容器（去掉外层，避免 pandoc 保留为原始 HTML）
    for figure in article.find_all("figure"):
        figure.unwrap()
    # 清空剩余 div 的 class/id/style 属性，让 pandoc 直接解包为普通容器
    for div in article.find_all("div"):
        for attr in list(div.attrs):
            del div[attr]

    # 4. 处理图片：只保留 light 变体，下载并改写 src 为本地相对路径，清掉多余属性
    img_count = 0
    for img in article.find_all("img"):
        classes = img.get("class") or []
        if "media-dark" in classes:
            img.decompose()  # 明暗双图只留 light 变体
            continue
        src = img.get("src", "")
        if not src:
            continue
        real_url = decode_next_image_url(src)
        if real_url.startswith("data:"):
            continue
        img_count += 1
        alt_text = img.get("alt", "") or ""
        fname = download_image(real_url, images_dir, img_count, alt_text)
        if fname:
            # 清空所有多余属性，只留 src/alt，保证 pandoc 能转成 Markdown 图片语法
            img.attrs = {"src": f"images/{fname}", "alt": alt_text}

    # 5. 标题清理：去掉文本里的 "#" 前缀符号（装饰用文本节点）
    for h in article.find_all(["h1", "h2", "h3", "h4"]):
        for node in h.find_all(string=True):
            if node.strip() == "#":
                node.extract()

    # 6. pandoc 转 Markdown
    tmp_html = out_dir / "_article.html"
    tmp_html.write_text(str(article), encoding="utf-8")
    md_path = out_dir / "mixture-of-kittens.md"
    result = subprocess.run(
        ["pandoc", str(tmp_html), "-f", "html", "-t", "gfm", "--wrap=none", "-o", str(md_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ pandoc 失败: {result.stderr}", file=sys.stderr)
        return 1
    tmp_html.unlink(missing_ok=True)

    # 7. 后处理：删除纯 <div>/</div> 包装行（无属性的空壳 div，pandoc 仍会保留）
    content = md_path.read_text(encoding="utf-8")
    cleaned_lines = [
        line for line in content.splitlines()
        if line.strip() not in ("<div>", "</div>", "<div >", "</div >")
    ]
    cleaned = "\n".join(cleaned_lines).strip() + "\n"

    # 8. 在 Markdown 开头补充元信息
    header = f"> 来源: https://cursor.com/cn/blog/mixture-of-kittens\n"
    if date_m:
        header += f"> 日期: {date_m.group(0)}\n"
    header += f"> 图片: {img_count} 张（已下载至 images/ 目录）\n\n"
    md_path.write_text(header + cleaned, encoding="utf-8")

    print(f"✅ 完成: {md_path}")
    print(f"✅ 图片: {img_count} 张 -> {images_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
