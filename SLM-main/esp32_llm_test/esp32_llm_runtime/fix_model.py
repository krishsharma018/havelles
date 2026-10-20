from pathlib import Path

p = Path("main/model/stories260K_model.cc")
s = p.read_text()

s = s.replace('extern "C" {', '')
s = s.replace('\n}\n', '\n')

p.write_text(s)
