import { readFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { resolve, relative, join, extname } from "node:path";
import { fileURLToPath } from "node:url";

// A whitelist is deliberate: an unreviewed Markdown file must stop deployment.
const root = fileURLToPath(new URL("../", import.meta.url));
const manifest = JSON.parse(readFileSync(join(root, "publication-manifest.json"), "utf8"));
const failures = [];
const forbidden = [
  /km\.sankuai\.com/,
  /求职|薪资|薪酬|招聘|简历|面试|人才缺口|职业规划|个人水平评估/,
  /\/Users\/|\/home\/qiker|target\.md|Learning-harness|M4 Pro/,
  /(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})/,
];

function filesUnder(directory) {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink()) {
      failures.push(`Symlink is not allowed in publication inputs: ${relative(root, path)}`);
      return [];
    }
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

function scan(path) {
  const text = readFileSync(path, "utf8");
  for (const pattern of forbidden) {
    if (pattern.test(text)) failures.push(`Sensitive-content pattern ${pattern} in ${relative(root, path)}`);
  }
  return text;
}

// Date prefixes are editorial names; approval follows the stable article stem.
const identity = (name) => name.replace(/(^|\/)\d{4}-\d{2}-\d{2}-/g, "$1");
const allowed = new Set(manifest.articles.map(identity));
const navigation = new Set(manifest.navigation);
const actual = new Map();
if (allowed.size !== manifest.articles.length) failures.push("Duplicate article in manifest");
for (const path of filesUnder(join(root, "content"))) {
  const name = relative(join(root, "content"), path).replaceAll("\\", "/");
  const key = identity(name);
  if (actual.has(key)) failures.push(`Duplicate article identity: ${name}`);
  actual.set(key, path);
  if (!allowed.has(key) && !navigation.has(name)) {
    failures.push(`Unreviewed content: ${name}`);
    continue;
  }
  const text = scan(path);
  if (allowed.has(key) && !/^visibility: public$/m.test(text)) {
    failures.push(`Article needs visibility: public: ${name}`);
  }
  if (/^draft: true$/m.test(text)) failures.push(`Manifest must not include drafts: ${name}`);
}
for (const name of [...allowed, ...navigation]) {
  const path = actual.get(name) || resolve(root, "content", name);
  if (!path.startsWith(resolve(root, "content") + "/") || !existsSync(path)) {
    failures.push(`Missing or invalid manifest path: ${name}`);
  }
}
for (const path of filesUnder(join(root, "static"))) {
  const name = relative(join(root, "static"), path).replaceAll("\\", "/");
  if (!manifest.assets.includes(name)) failures.push(`Unreviewed public asset: ${name}`);
}

const outputIndex = process.argv.indexOf("--output");
if (outputIndex >= 0) {
  const output = resolve(process.argv[outputIndex + 1] || "public");
  if (!existsSync(join(output, "index.html"))) failures.push("Build output is missing index.html");
  const outputFiles = filesUnder(output);
  const htmlFiles = outputFiles.filter((path) => path.endsWith(".html"));
  const origin = new URL(process.env.SITE_BASE_URL || manifest.baseURL);
  for (const path of outputFiles) {
    if ([".html", ".xml", ".json", ".txt"].includes(extname(path))) scan(path);
    if (/\.(?:pdf|zip|mp4|mov)$/i.test(path)) failures.push(`Unexpected archive or media in build: ${path}`);
    if (relative(output, path).split(/[\\/]/).some((part) => ["career", "private", "research"].includes(part))) {
      failures.push(`Non-public route in build: ${path}`);
    }
  }
  let links = 0;
  for (const path of htmlFiles) {
    const text = readFileSync(path, "utf8");
    const documentURL = new URL(relative(output, path), origin);
    // Hugo's minifier may remove quotes around attribute values.
    for (const match of text.matchAll(/(?:href|src)=(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g)) {
      const target = match[1] ?? match[2] ?? match[3];
      if (target.startsWith("#") || target.startsWith("data:")) continue;
      const url = new URL(target.replaceAll("&amp;", "&"), documentURL);
      if (url.origin !== origin.origin) continue;
      if (!url.pathname.startsWith(origin.pathname)) {
        failures.push(`Link escapes Pages base path: ${target}`);
        continue;
      }
      const destination = join(output, decodeURIComponent(url.pathname.slice(origin.pathname.length)));
      if (!existsSync(destination) || (statSync(destination).isDirectory() && !existsSync(join(destination, "index.html")))) {
        failures.push(`Broken local link: ${relative(output, path)} -> ${target}`);
      }
      links += 1;
    }
  }
  console.log(`Checked ${htmlFiles.length} HTML pages, ${links} local links; scanned HTML/RSS/JSON/text.`);
}
if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log(`Publication check passed: ${allowed.size} approved articles.`);
}
