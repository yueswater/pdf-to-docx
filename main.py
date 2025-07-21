import logging
import os
import shutil
import time

from pdf_to_docx.ocr.processor import PDFTextExtractor
from pdf_to_docx.pipeline.md2docx import DocxExporter
from pdf_to_docx.pipeline.txt2md import TextToMarkdown
from pdf_to_docx.utils import make_output_dirs
from pdf_to_docx.utils import print_stat
from pdf_to_docx.utils.logging_tools.logger import setup_logger
from tqdm import tqdm

setup_logger()

if __name__ == "__main__":
    # timer
    start = time.time()

    # define folder
    DATA_DIR = "./data/"
    OUTPUT_DIR, OUTPUT_DIRs = "./output", ["tx", "pdf", "md", "doc"]
    folder = make_output_dirs(OUTPUT_DIR, OUTPUT_DIRs)
    TXT_DIR, PDF_DIR, MD_DIR, DOC_DIR = (
        folder["tx"],
        folder["pdf"],
        folder["md"],
        folder["doc"],
    )

    # process all PDF files
    file_lst = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    success = 0
    total = len(file_lst)
    logging.debug(f"Now processing {total} file(s) in {DATA_DIR}")

    for filename in tqdm(file_lst, desc="Processing PDF files"):
        pdf_path = os.path.join(DATA_DIR, filename)
        txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))
        md_path = os.path.join(MD_DIR, filename.replace(".pdf", ".md"))
        docx_path = os.path.join(DOC_DIR, filename.replace(".pdf", ".docx"))
        pdf_out_path = os.path.join(
            PDF_DIR, filename.replace(".pdf", "(annotated).pdf")
        )

        try:
            extractor = PDFTextExtractor(pdf_path)
            text_list = extractor()
            logging.debug(f"Open PDF file {pdf_path}")

            if text_list:
                full_text = "\n\n".join(text_list)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                    # for line in text_list:
                    #     f.write(line.strip() + "\n\n")

                # page number box (by PDF or OCR mode)
                if extractor.pagenum_blocks:
                    extractor.draw_pagenum_boxes(pdf_out_path)
                    logging.debug(
                        f"PDF mode: page number block found?"
                        f"{bool(extractor.pagenum_blocks)}"
                    )
                elif extractor.ocr_page_number_boxes:
                    logging.debug(
                        f"OCR mode: page number block found?"
                        f"{bool(extractor.ocr_page_number_boxes)}"
                    )
                    extractor.draw_ocr_pagenum_boxes(pdf_out_path)
                else:
                    logging.debug(f"No page number block found (PDF/OCR): {filename}")

            if os.path.exists(extractor.temp_path):
                shutil.rmtree(extractor.temp_path)

            # convert to markdown
            md_converter = TextToMarkdown.from_text(text_list, txt_path, md_path)
            md_converter.save()
            logging.debug(f"Saved markdown file in {md_path}")

            # convert to Word file
            docx_converter = DocxExporter(
                md_path=md_path, output_dir=MD_DIR, target_dir=DOC_DIR
            )
            docx_converter.export_docx()
            logging.debug(f"Saved Word file in {docx_path}")

            success += 1

        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")

    # statistical results
    print_stat(start, total, success)
