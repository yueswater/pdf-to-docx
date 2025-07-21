import json
import logging
import shutil
from pathlib import Path

from pdf_to_docx.ocr.processor import PDFTextExtractor
from pdf_to_docx.pipeline.md2docx import DocxExporter
from pdf_to_docx.pipeline.mdconvert import TextToMarkdownAI
from pdf_to_docx.pipeline.txt2md import TextToMarkdown

def process_pdf_file(
    pdf_path: Path,
    txt_path: Path,
    md_path: Path,
    docx_path: Path,
    json_path: Path,
    pdf_out_path: Path,
    use_openai: bool = False,
    debug: bool = False,
):
    pdf_path, txt_path, md_path, docx_path, json_path, pdf_out_path = map(
        Path, (pdf_path, txt_path, md_path, docx_path, json_path, pdf_out_path)
    )

    # Generate paths
    pdf_path = Path(pdf_path)

    # Mode saver
    results = {"success": None, "mode": None}

    # Extract full text from PDF
    try:
        # Define extractor
        extractor = PDFTextExtractor(
            pdf_path, auto_clean=True, disable_tqdm=True, debug=debug
        )
        text_list = extractor()
        logging.debug(f"Open PDF file {pdf_path}")

        if text_list:
            with open(txt_path, "w", encoding="utf-8") as f:
                for line in text_list:
                    f.write(line.strip() + "\n\n")

            results["mode"] = extractor.mode

            # Draw page number box (by PDF or OCR mode)
            if extractor.pagenum_blocks:
                extractor.draw_pagenum_boxes(pdf_out_path)
                logging.debug(
                    "PDF mode: page number block found: %s",
                    extractor.pagenum_blocks is not None,
                )
            elif extractor.ocr_page_number_boxes:
                logging.debug(
                    "OCR mode: page number block found: %s",
                    extractor.ocr_page_number_boxes is not None,
                )
                extractor.draw_ocr_pagenum_boxes(pdf_out_path)
            else:
                logging.warning("No page number block found (PDF/OCR): %s", pdf_path)
                extractor.original_pdf(pdf_out_path)
        else:
            logging.warning("Empty result when parsing PDF file: %s", pdf_path)
            return results

        if Path(extractor.temp_path).exists():
            shutil.rmtree(extractor.temp_path)

        # Save page numbers to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            page_info = extractor.get_page_info()
            json.dump(page_info, f, indent=4, ensure_ascii=False)
            logging.debug(f"Saved page info JSON file in {json_path}")

        # Convert to markdown
        if use_openai:
            converter = TextToMarkdownAI(txt_path, md_path)
            converter.run()
            logging.debug(f"Saved markdown file in {md_path} using OpenAI")
        else:
            converter = TextToMarkdown.from_text(text_list, txt_path, md_path)
            converter.save()
            logging.debug(f"Saved markdown file in {md_path} using TextToMarkdown")

        # Convert to Word file
        docx_converter = DocxExporter(
            md_path=md_path, output_dir=md_path.parent, target_dir=docx_path.parent
        )
        docx_converter.export_docx()
        logging.debug(f"Saved Word file in {docx_path}")

        results["success"] = True
        return results

    except Exception as e:
        logging.exception(f"Failed to process {pdf_path.stem}: {e}")
        results["success"] = False
        return results
