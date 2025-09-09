"""Cache serialization utilities."""

import json
import pickle
import gzip
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class CacheSerializer:
    """Serializer for cache data with compression and encryption support."""
    
    def __init__(self, enable_compression: bool = True, enable_encryption: bool = False):
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        if enable_encryption:
            try:
                from cryptography.fernet import Fernet
                self.fernet = Fernet
                # In production, this should be loaded from secure config
                self.encryption_key = Fernet.generate_key()
                self.cipher = Fernet(self.encryption_key)
            except ImportError:
                logger.warning("cryptography not available, encryption disabled")
                self.enable_encryption = False
                self.cipher = None
        else:
            self.cipher = None
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        try:
            # Pickle the data
            serialized = pickle.dumps(data)
            
            # Compress if enabled
            if self.enable_compression:
                serialized = gzip.compress(serialized)
            
            # Encrypt if enabled
            if self.enable_encryption and self.cipher:
                serialized = self.cipher.encrypt(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            # Fallback to JSON
            return json.dumps(data).encode()
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        try:
            # Decrypt if enabled
            if self.enable_encryption and self.cipher:
                data = self.cipher.decrypt(data)
            
            # Decompress if enabled
            if self.enable_compression:
                data = gzip.decompress(data)
            
            # Unpickle the data
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            # Fallback to JSON
            try:
                return json.loads(data.decode())
            except Exception as e2:
                logger.error(f"JSON fallback also failed: {e2}")
                return None
