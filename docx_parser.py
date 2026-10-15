from docx import Document


def extract_text_from_docx(file_path):
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text