import argparse
import logging
from pathlib import Path

from pdf_to_docx.main import process_pdf_file
from pdf_to_docx.utils.io_tools.make_output_dirs import make_output_dirs
from pdf_to_docx.utils.logging_tools.logger import setup_logger, LoggerStatus
from pdf_to_docx.utils.time_tools.measure_time import measure_time

setup_logger(LoggerStatus.FULL)

@measure_time
def main():
    """
    Command-line tool for converting a PDF file to OCR-processed Markdown and Word documents.

    This script supports both local OCR and OpenAI-based conversion. It outputs cleaned and
    structured results in multiple formats including TXT, Markdown (.md), Word (.docx), and
    an annotated version of the original PDF.

    Arguments:
        pdf_path (str): Path to the input PDF file (required).
        -o, --output (str): Output directory (default: ./output).
        -v, --verbose (int): Verbosity level. 0 = silent, 1 = info (default), 2 = debug.
        --openai: Use local OCR (default). Use this flag to disable OpenAI-based processing.

    Example usage:
        python cli.py ./input/sample.pdf
        python cli.py ./input/sample.pdf -o ./results
        python cli.py ./input/sample.pdf --openai
        python cli.py ./input/sample.pdf -v 2 -o ./out --openai

    Output folders created:
        - <output>/tx   : Raw OCR text
        - <output>/md   : Cleaned Markdown
        - <output>/doc  : Word (.docx) document
        - <output>/pdf  : Annotated PDF (optional)

    Returns:
        None
    """
    # Initialize CLI Arg Parser
    parser = argparse.ArgumentParser(
        description="PDF to OCR Markdown/Word",
        # formatter_class=RichHelpFormatter,
    )

    # Add arguments
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Show page number detected information",
    )
    parser.add_argument(
        "--openai",
        dest="use_openai",
        action="store_true",
        help="Use OpenAI instead of local OCR converter (default is local OCR converter)",
    )
    parser.set_defaults(use_openai=False)

    # Collects arguments
    args = parser.parse_args()

    # Make output folders & define output file name
    folder = make_output_dirs(args.output, ["tx", "pdf", "md", "doc", "json"])
    pdf_path = Path(args.pdf_path)
    filename = pdf_path.stem

    pdf_convert = process_pdf_file(
        pdf_path=Path(args.pdf_path),
        txt_path=folder["tx"] / Path(filename).with_suffix(".txt"),
        md_path=folder["md"] / Path(filename).with_suffix(".md"),
        docx_path=folder["doc"] / Path(filename).with_suffix(".docx"),
        json_path=folder["json"] / Path(filename).with_suffix(".json"),
        pdf_out_path=folder["pdf"] / f"{pdf_path.stem}(annotated).pdf",
        use_openai=args.use_openai,
        debug=args.debug,
    )

    if pdf_convert:
        logging.debug(f"Successfully convert {pdf_path} to Markdown, JSON & Word file!")
    else:
        logging.debug(f"Failed processing {pdf_path}")


if __name__ == "__main__":
    main()
