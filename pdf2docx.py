import argparse
import logging
import os

from pdf_to_docx.main import process_pdf_file
from pdf_to_docx.utils.io_tools.make_output_dirs import make_output_dirs
from pdf_to_docx.utils.logging_tools.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="PDF to OCR Markdown/Word")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level: 0 = silent, 1 = info (default), 2 = debug",
    )
    parser.add_argument(
        "--no-openai",
        dest="use_openai",
        action="store_false",
        help="Use local OCR converter instead of OpenAI (default is to use OpenAI)",
    )
    parser.set_defaults(use_openai=True)

    args = parser.parse_args()

    if args.verbose == 0:
        logging_level = logging.CRITICAL
    elif args.verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG

    setup_logger(level=logging_level)

    folder = make_output_dirs(args.output, ["tx", "pdf", "md", "doc"])
    filename = os.path.basename(args.pdf_path).replace(".pdf", "")

    success = process_pdf_file(
        pdf_path=args.pdf_path,
        txt_path=os.path.join(folder["tx"], filename + ".txt"),
        md_path=os.path.join(folder["md"], filename + ".md"),
        docx_path=os.path.join(folder["doc"], filename + ".docx"),
        pdf_out_path=os.path.join(folder["pdf"], filename + "(annotated).pdf"),
        use_openai=args.use_openai,
    )

    if success:
        logging.debug(f"Successfully convert {filename} to Markdown & Word file!")
    else:
        logging.debug(f"Failed processing {filename}")


if __name__ == "__main__":
    main()
