"""index.html 结构校验：标签配对 + 锚点可达 + 关键内容存在。"""
import re
import sys
from html.parser import HTMLParser

SRC = "index.html"
html = open(SRC, encoding="utf-8").read()

VOID = {"meta", "link", "br", "img", "input", "hr", "source"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))
        if tag == "script" or tag == "style":
            self.stack.append(("RAW:" + tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        want = tag if not tag.startswith("RAW") else tag
        # RAW 栀记由 set_cdata_mode 处理，这里只做普通配对
        for i in range(len(self.stack) - 1, -1, -1):
            t, pos = self.stack[i]
            if t == want or t == "RAW:" + want:
                del self.stack[i:]
                return
        self.errors.append(f"orphan </{tag}> at {self.getpos()}")


c = Checker()
c.feed(html)
c.close()
for t, pos in c.stack:
    if not t.startswith("RAW:"):
        c.errors.append(f"unclosed <{t}> at {pos}")

anchors = set(re.findall(r'href="#([\w-]+)"', html))
ids = set(re.findall(r'id="([\w-]+)"', html))
missing = anchors - ids
if missing:
    c.errors.append(f"dangling anchors: {sorted(missing)}")

must_have = [
    "八类灾种监测", "基准矩阵", "id=\"benchmark\"", "62%", "0%",
    "郑州 7·20", "杜苏芮", "通辽寒潮", "重庆高温", "FWI 70.5",
    "GB/T 20484-2017", "32.7", "24.5", "llm-eval-carm.md",
    "historical-validation.md", "Qwen3.6-35B",
]
for token in must_have:
    if token not in html:
        c.errors.append(f"missing content token: {token}")

forbidden = ["四大灾种", "Ollama", "Qwen3 14B", "全球首个", "\ufffd", "20 个测试"]
for token in forbidden:
    if token in html:
        c.errors.append(f"stale token still present: {token}")

dup_ids = [i for i in ids
           if len(re.findall(f'id="{i}"', html)) > 1 and i != "RAW"]
if dup_ids:
    c.errors.append(f"duplicate ids: {sorted(set(dup_ids))}")

if c.errors:
    print("FAIL")
    for e in c.errors[:20]:
        print(" -", e)
    sys.exit(1)
print(f"OK: tags balanced, anchors {sorted(anchors)} all resolve, "
      f"{len(must_have)} content tokens present, no stale tokens")
