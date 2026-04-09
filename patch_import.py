lines = open("src/lavello_mlips/process_omol25.py").read().splitlines()

s3_idx = lines.index("from .s3_processor import S3DataProcessor")
lines.pop(s3_idx)

util_idx = lines.index("from ase.parallel import DummyMPI")
lines.insert(util_idx + 1, "from .s3_processor import S3DataProcessor")

open("src/lavello_mlips/process_omol25.py", "w").write("\n".join(lines) + "\n")
