from pathlib import Path

data = Path("main/model/stories260K.bin").read_bytes()

with open("main/model/stories260K_model.cc", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write('extern "C" {\n')
    f.write("const unsigned char stories260K_model[] = {\n")

    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        f.write("    " + ",".join(f"0x{x:02x}" for x in chunk) + ",\n")

    f.write("};\n")
    f.write(f"const unsigned int stories260K_model_len = {len(data)};\n")
    f.write("}\n")
