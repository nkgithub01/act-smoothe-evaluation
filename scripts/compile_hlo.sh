backend=$1

mkdir -p asm
/workspace/act-backends/${backend} --input attention.hlo --output /workspace/asm/compiled_${backend}.py