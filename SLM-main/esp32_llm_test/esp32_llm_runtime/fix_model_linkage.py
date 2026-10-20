from pathlib import Path

p = Path("main/model/stories260K_model.cc")
s = p.read_text()

s = s.replace(
    "const unsigned char stories260K_model[] = {",
    "extern const unsigned char stories260K_model[] = {",
    1
)

s = s.replace(
    "const unsigned int stories260K_model_len = 1056540;",
    "extern const unsigned int stories260K_model_len = 1056540;",
    1
)

p.write_text(s)
