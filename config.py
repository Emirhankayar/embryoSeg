from pathlib import Path

embryo_base_path = "/home/phd2/Scrivania/CorsoData/blastocisti"
# embryo_base_path = "/run/media/capitan/Emu/blastodata_orig"
doc_base_path = "/home/phd2/Documenti/embryoSeg"
csv_path = Path(doc_base_path) / "bbox.csv"
model_path = Path(doc_base_path) / ""

blasto_dir_label = "blasto"
noblasto_dir_label = "no_blasto"
