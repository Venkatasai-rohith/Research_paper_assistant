class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""

class VectorStoreCreationError(Exception):
    """Raised when Vectorstore creation fails."""

class VectorStoreSaveError(Exception):
    """Raised while saving the vectorstore fails."""

class VectorStoreLoadError(Exception):
    """Raised when loading a vectorstore fails."""

class PaperNotFoundError(Exception):
    """Raised when paper loading fails."""