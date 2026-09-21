# 拾阶 · 技术笔记

基于 [Hugo PaperMod](https://github.com/adityatelange/hugo-PaperMod) 的中文技术博客，面向 GitHub Pages 发布。首页为 Infra、算法、技术专题三个阶梯状入口，支持手机布局、公式、深浅色、全文搜索、归档和 RSS。

## 内容与可见性

公开文章包括技术整理、带来源的阅读导引和图文研讨；当前收录以 `publication-manifest.json` 为准。按系统架构、推理引擎、缓存与存储、算子与通信、注意力算法、训练基础设施、模型架构、模型与评测分类。

私人原稿保存在博客仓库之外。不要把敏感内容提交到此仓库，即使标为 draft 或从首页隐藏，也仍可能从公开 Git 历史中读到。

`publication-manifest.json` 明确列出可发布正文、导航与静态资源。Pages 工作流在构建前检查白名单与敏感模式，构建后再检查页面、搜索 JSON、RSS 和站内链接。新文件不在白名单时发布会失败；自动扫描只能辅助审核，不能证明文本一定不含私人信息。

整理不等于再次验证原文：文章说明了原记录时间、证据等级、硬件前提及未复验范围。阅读导引按来源日期排序，并标注 2026-09-21 的实际整理日期；9 月材料不倒填到更早的月份。

文章文件可添加或调整 `YYYY-MM-DD-` 前缀，白名单按去掉此前缀的文件名识别；站内文章链接使用 `{{< article "infra/article-name.md" >}}`，自动解析当前文件路径。同名文章会阻止构建，新增主题仍须人工审核加入白名单。

## 本地预览

导航中的「图文研讨」收录两篇 `session/` 图文文章（公开编辑版在 `content/session/`）。51 个图片与 GIF 保存在 `static/media/session/`，按内容摘要命名，避免中文、空格及百分号编码导致路径错误。Markdown 使用 `media/session/文件名`，图片渲染模板自动补齐 Pages 子路径；支持自适应宽度、懒加载和点击原图，GIF 不转为静态缩略图。原始 `../session/` 不会自动发布，后续编辑需要同步公开版并审核。正文中的内网视频附件链接不公开，保留对应 GIF。

生产构建后运行 `node scripts/check-session-images.mjs public` 验证 51 张图的引用、原始字节及 GIF 循环信息；Pages 工作流也会执行这项检查。素材版权与正文事实仍需在正式公开前由作者确认。

安装 Hugo **0.166.0**，然后运行：

```sh
git submodule update --init --recursive
hugo server
```

打开终端显示的本地地址。生产检查：

```sh
node scripts/check-publication.mjs
hugo --gc --minify --cleanDestinationDir
node scripts/check-publication.mjs --output public
```

需要 Node.js 24，检查器无第三方依赖。公式使用固定版本的 MathJax CDN；公式文本本身保留在 Markdown 中，CDN 不可达时仍可阅读源码表示。

## 写文章

```sh
hugo new content infra/my-first-note.md
# 或 algorithms/my-first-note.md
```

补全标题、description、categories 和正文，人工确认适合公开后，将 `draft` 改为 `false`、`visibility` 改为 `public`，把相对 `content/` 的文章路径加入白名单，再运行上述检查。两篇初始学习导读的日期为展示日期，不代表历史发布记录；Git 提交时间保持真实。

## GitHub Pages

1. 把这个独立仓库推送到自己的 GitHub 仓库，默认分支为 `main`。
2. 在 GitHub 仓库的 **Settings → Pages → Source** 选择 **GitHub Actions**。
3. 推送 `main`，或在 Actions 中手动运行 `Deploy blog to GitHub Pages`。
4. 等待部署成功后访问 Pages 给出的地址。默认配置为 `https://ckqqqq.github.io/blog/`。

工作流自动从 Pages 读取部署 URL，支持项目子路径。更改仓库名或使用自定义域名时，同步修改 `hugo.yaml` 和 `publication-manifest.json` 的 `baseURL`，以便本地生产检查保持一致。本次仅准备发布配置，未实际上传或上线。

## 修改首页

- `hugo.yaml`：站点名称、作者、菜单。
- `layouts/home.html`：首页文案、三个入口及最近文章。
- `assets/css/extended/journal.css`：配色、阶梯布局与手机适配。
- `content/`：文章和专题介绍。

PaperMod 以 Git submodule 引入并锁定版本，许可证保留在 `themes/PaperMod/LICENSE`。本仓库只收录经过整理的公开技术版本，不包含完整的私人研究目录、PDF 存档或视频素材。
