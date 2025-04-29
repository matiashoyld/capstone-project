#!/usr/bin/env python3
"""
PDF to TXT Converter

This script converts all PDF files in the draft/papers/pdf directory to text files
and saves them in the draft/papers/txt directory.
"""

import os
import sys
import argparse
from pathlib import Path
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2
    
    Args:
        pdf_path (Path): Path to the PDF file
    
    Returns:
        str: Extracted text
    """
    text = ""
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
                text += "\n\n--- Page Break ---\n\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def convert_pdfs_to_txt(pdf_dir, txt_dir, overwrite=False):
    """
    Convert all PDFs in a directory to text files
    
    Args:
        pdf_dir (Path): Directory containing PDF files
        txt_dir (Path): Directory to save text files
        overwrite (bool): Whether to overwrite existing text files
    """
    # Ensure txt directory exists
    txt_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to convert")
    
    # Convert each PDF file to text
    for i, pdf_path in enumerate(pdf_files, 1):
        # Create output path
        txt_path = txt_dir / f"{pdf_path.stem}.txt"
        
        # Check if output file already exists
        if txt_path.exists() and not overwrite:
            print(f"[{i}/{total_files}] Skipping {pdf_path.name} (already exists)")
            continue
        
        print(f"[{i}/{total_files}] Converting {pdf_path.name}...", end="", flush=True)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Save text to file
        if text:
            try:
                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(text)
                print(" Done!")
            except Exception as e:
                print(f" Error writing to {txt_path}: {e}")
        else:
            print(" Failed to extract text!")

def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to text files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing text files")
    args = parser.parse_args()
    
    # Define directories
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "pdf"
    txt_dir = script_dir / "txt"
    
    # Check if pdf directory exists
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1
    
    # Convert PDFs to text
    convert_pdfs_to_txt(pdf_dir, txt_dir, args.overwrite)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
