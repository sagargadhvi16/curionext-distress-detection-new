"""Dataset validation script for integrity, format, and sample count checks."""
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
import logging

# Try importing audio libraries (optional)
try:
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    dataset_name: str
    status: str  # 'valid', 'invalid', 'partial', 'missing'
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    missing_files: int = 0
    expected_count: Optional[int] = None
    actual_count: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    file_integrity_checks: Dict[str, bool] = None
    format_checks: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.file_integrity_checks is None:
            self.file_integrity_checks = {}
        if self.format_checks is None:
            self.format_checks = {}


class DatasetValidator:
    """Validator for all datasets."""
    
    # Expected dataset structures and counts
    DATASET_EXPECTATIONS = {
        'esc50': {
            'path': 'data/raw/audio/esc50',
            'expected_count': 2000,
            'file_extensions': ['.wav'],
            'structure': 'category_based',
            'categories': 50  # ESC-50 has 50 categories
        },
        'ravdess': {
            'path': 'data/raw/audio/ravdess',
            'expected_count': None,  # Varies by dataset version
            'file_extensions': ['.wav'],
            'structure': 'emotion_based',
            'expected_emotions': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        },
        'audioset': {
            'path': 'data/raw/audio/audioset',
            'expected_count': None,  # Varies
            'file_extensions': ['.wav', '.mp3'],
            'structure': 'category_based',
            'categories': ['crying', 'screaming', 'human_sounds']
        },
        'wesad': {
            'path': 'data/raw/biometric/wesad',
            'expected_count': None,
            'file_extensions': ['.pkl', '.mat', '.json'],
            'structure': 'subject_based'
        },
        'pamap2': {
            'path': 'data/raw/biometric/pamap2',
            'expected_count': None,
            'file_extensions': ['.dat', '.csv'],
            'structure': 'subject_based'
        }
    }
    
    def __init__(self, base_path: Path = None, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Args:
            base_path: Base project path (defaults to script parent)
            logger: Logger instance (creates one if not provided)
        """
        self.base_path = base_path or project_root
        self.logger = logger or setup_logger(__name__, log_file='dataset_validation.log', level=logging.INFO)
        self.results: Dict[str, ValidationResult] = {}
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'md5') -> Optional[str]:
        """
        Calculate file hash for integrity checking.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha256')
            
        Returns:
            Hex digest of file hash, or None if error
        """
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def validate_audio_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate audio file format and integrity.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check file exists
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors
        
        # Check file is not empty
        if file_path.stat().st_size == 0:
            errors.append(f"File is empty: {file_path}")
            return False, errors
        
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        if file_path.suffix.lower() not in valid_extensions:
            errors.append(f"Invalid file extension: {file_path.suffix}")
            # Don't return False here, just warn
        
        # Try to load with librosa if available
        if AUDIO_LIBS_AVAILABLE:
            try:
                y, sr = librosa.load(str(file_path), sr=None, duration=1.0)
                if len(y) == 0:
                    errors.append(f"Audio file has no samples: {file_path}")
                    return False, errors
                if sr < 1000:  # Very low sample rate is suspicious
                    errors.append(f"Unusually low sample rate: {sr} Hz")
            except Exception as e:
                errors.append(f"Failed to load audio file: {str(e)}")
                return False, errors
        
        # Try to read with soundfile if available (for WAV)
        elif SOUNDFILE_AVAILABLE and file_path.suffix.lower() == '.wav':
            try:
                data, samplerate = sf.read(str(file_path))
                if len(data) == 0:
                    errors.append(f"Audio file has no samples: {file_path}")
                    return False, errors
            except Exception as e:
                errors.append(f"Failed to read WAV file: {str(e)}")
                return False, errors
        
        return len(errors) == 0, errors
    
    def validate_json_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate JSON file format.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors
        
        if file_path.stat().st_size == 0:
            errors.append(f"File is empty: {file_path}")
            return False, errors
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading JSON file: {str(e)}")
            return False, errors
        
        return True, errors
    
    def validate_csv_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate CSV file format (basic check).
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors
        
        if file_path.stat().st_size == 0:
            errors.append(f"File is empty: {file_path}")
            return False, errors
        
        # Basic check: try to read first few lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline() for _ in range(3)]
                if not any(lines):
                    errors.append("CSV file appears to be empty")
        except Exception as e:
            errors.append(f"Error reading CSV file: {str(e)}")
            return False, errors
        
        return True, errors
    
    def validate_file_format(self, file_path: Path, expected_extensions: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate file format based on extension.
        
        Args:
            file_path: Path to file
            expected_extensions: List of expected extensions (e.g., ['.wav', '.mp3'])
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not file_path.exists():
            return False, [f"File does not exist: {file_path}"]
        
        ext = file_path.suffix.lower()
        
        if ext in ['.wav', '.mp3', '.flac', '.ogg']:
            return self.validate_audio_file(file_path)
        elif ext == '.json':
            return self.validate_json_file(file_path)
        elif ext in ['.csv', '.dat']:
            return self.validate_csv_file(file_path)
        elif ext in ['.pkl', '.mat']:
            # For pickle/matlab files, just check existence and non-empty
            if file_path.stat().st_size == 0:
                return False, [f"File is empty: {file_path}"]
            return True, []
        else:
            # Unknown format, just check existence
            self.logger.warning(f"Unknown file format: {ext} for {file_path}")
            return True, []
    
    def count_files_in_directory(self, directory: Path, extensions: List[str], recursive: bool = True) -> int:
        """
        Count files with specified extensions in directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions (with dots, e.g., ['.wav', '.mp3'])
            recursive: Search recursively
            
        Returns:
            Number of matching files
        """
        if not directory.exists():
            return 0
        
        count = 0
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.rglob(pattern) if recursive else directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in [e.lower() for e in extensions]:
                count += 1
        
        return count
    
    def validate_esc50(self) -> ValidationResult:
        """Validate ESC-50 dataset."""
        self.logger.info("Validating ESC-50 dataset...")
        result = ValidationResult(
            dataset_name='ESC-50',
            status='missing',
            expected_count=self.DATASET_EXPECTATIONS['esc50']['expected_count']
        )
        
        dataset_path = self.base_path / self.DATASET_EXPECTATIONS['esc50']['path']
        
        if not dataset_path.exists():
            result.errors.append(f"Dataset directory does not exist: {dataset_path}")
            result.status = 'missing'
            return result
        
        # Count audio files
        extensions = self.DATASET_EXPECTATIONS['esc50']['file_extensions']
        result.actual_count = self.count_files_in_directory(dataset_path, extensions, recursive=True)
        result.total_files = result.actual_count
        
        if result.actual_count == 0:
            result.errors.append("No audio files found in dataset directory")
            result.status = 'missing'
            return result
        
        # Validate sample files
        valid_count = 0
        invalid_count = 0
        
        # Sample validation (check up to 100 files to avoid long runtime)
        sample_files = list(dataset_path.rglob('*.wav'))[:100]
        
        for file_path in sample_files:
            is_valid, errors = self.validate_audio_file(file_path)
            if is_valid:
                valid_count += 1
                result.file_integrity_checks[str(file_path)] = True
            else:
                invalid_count += 1
                result.file_integrity_checks[str(file_path)] = False
                result.errors.extend(errors[:1])  # Add first error
        
        # Estimate total valid based on sample
        if len(sample_files) > 0:
            valid_ratio = valid_count / len(sample_files)
            result.valid_files = int(result.actual_count * valid_ratio)
            result.invalid_files = result.actual_count - result.valid_files
        else:
            result.valid_files = result.actual_count
            result.invalid_files = 0
        
        # Determine status
        if result.actual_count >= result.expected_count * 0.9:  # 90% threshold
            if result.invalid_files == 0:
                result.status = 'valid'
            else:
                result.status = 'partial'
        elif result.actual_count > 0:
            result.status = 'partial'
            result.warnings.append(f"Only {result.actual_count}/{result.expected_count} files found")
        else:
            result.status = 'missing'
        
        return result
    
    def validate_ravdess(self) -> ValidationResult:
        """Validate RAVDESS dataset."""
        self.logger.info("Validating RAVDESS dataset...")
        result = ValidationResult(
            dataset_name='RAVDESS',
            status='missing'
        )
        
        dataset_path = self.base_path / self.DATASET_EXPECTATIONS['ravdess']['path']
        
        if not dataset_path.exists():
            result.errors.append(f"Dataset directory does not exist: {dataset_path}")
            result.status = 'missing'
            return result
        
        extensions = self.DATASET_EXPECTATIONS['ravdess']['file_extensions']
        result.actual_count = self.count_files_in_directory(dataset_path, extensions, recursive=True)
        result.total_files = result.actual_count
        
        if result.actual_count == 0:
            result.errors.append("No audio files found in dataset directory")
            result.status = 'missing'
            return result
        
        # Sample validation
        sample_files = list(dataset_path.rglob('*.wav'))[:50]
        valid_count = 0
        
        for file_path in sample_files:
            is_valid, _ = self.validate_audio_file(file_path)
            if is_valid:
                valid_count += 1
                result.file_integrity_checks[str(file_path)] = True
            else:
                result.file_integrity_checks[str(file_path)] = False
        
        if len(sample_files) > 0:
            valid_ratio = valid_count / len(sample_files)
            result.valid_files = int(result.actual_count * valid_ratio)
            result.invalid_files = result.actual_count - result.valid_files
        else:
            result.valid_files = result.actual_count
            result.invalid_files = 0
        
        result.status = 'valid' if result.invalid_files == 0 else 'partial'
        
        return result
    
    def validate_audioset(self) -> ValidationResult:
        """Validate AudioSet subset dataset."""
        self.logger.info("Validating AudioSet dataset...")
        result = ValidationResult(
            dataset_name='AudioSet',
            status='missing'
        )
        
        dataset_path = self.base_path / self.DATASET_EXPECTATIONS['audioset']['path']
        
        if not dataset_path.exists():
            result.errors.append(f"Dataset directory does not exist: {dataset_path}")
            result.status = 'missing'
            return result
        
        extensions = self.DATASET_EXPECTATIONS['audioset']['file_extensions']
        result.actual_count = self.count_files_in_directory(dataset_path, extensions, recursive=True)
        result.total_files = result.actual_count
        
        if result.actual_count == 0:
            result.errors.append("No audio files found in dataset directory")
            result.status = 'missing'
            return result
        
        # Sample validation
        sample_files = list(dataset_path.rglob('*'))[:50]
        sample_files = [f for f in sample_files if f.is_file() and f.suffix.lower() in ['.wav', '.mp3']]
        
        valid_count = 0
        for file_path in sample_files:
            is_valid, _ = self.validate_audio_file(file_path)
            if is_valid:
                valid_count += 1
                result.file_integrity_checks[str(file_path)] = True
            else:
                result.file_integrity_checks[str(file_path)] = False
        
        if len(sample_files) > 0:
            valid_ratio = valid_count / len(sample_files)
            result.valid_files = int(result.actual_count * valid_ratio)
            result.invalid_files = result.actual_count - result.valid_files
        else:
            result.valid_files = result.actual_count
            result.invalid_files = 0
        
        result.status = 'valid' if result.invalid_files == 0 else 'partial'
        
        return result
    
    def validate_wesad(self) -> ValidationResult:
        """Validate WESAD dataset."""
        self.logger.info("Validating WESAD dataset...")
        result = ValidationResult(
            dataset_name='WESAD',
            status='missing'
        )
        
        dataset_path = self.base_path / self.DATASET_EXPECTATIONS['wesad']['path']
        
        if not dataset_path.exists():
            result.errors.append(f"Dataset directory does not exist: {dataset_path}")
            result.status = 'missing'
            return result
        
        extensions = self.DATASET_EXPECTATIONS['wesad']['file_extensions']
        result.actual_count = self.count_files_in_directory(dataset_path, extensions, recursive=True)
        result.total_files = result.actual_count
        
        if result.actual_count == 0:
            result.errors.append("No data files found in dataset directory")
            result.status = 'missing'
            return result
        
        # Sample validation
        sample_files = list(dataset_path.rglob('*'))[:20]
        sample_files = [f for f in sample_files if f.is_file() and f.suffix.lower() in extensions]
        
        valid_count = 0
        for file_path in sample_files:
            if file_path.suffix.lower() == '.json':
                is_valid, _ = self.validate_json_file(file_path)
            else:
                is_valid = file_path.stat().st_size > 0
            
            if is_valid:
                valid_count += 1
                result.file_integrity_checks[str(file_path)] = True
            else:
                result.file_integrity_checks[str(file_path)] = False
        
        if len(sample_files) > 0:
            valid_ratio = valid_count / len(sample_files)
            result.valid_files = int(result.actual_count * valid_ratio)
            result.invalid_files = result.actual_count - result.valid_files
        else:
            result.valid_files = result.actual_count
            result.invalid_files = 0
        
        result.status = 'valid' if result.invalid_files == 0 and result.actual_count > 0 else 'partial'
        
        return result
    
    def validate_pamap2(self) -> ValidationResult:
        """Validate PAMAP2 dataset."""
        self.logger.info("Validating PAMAP2 dataset...")
        result = ValidationResult(
            dataset_name='PAMAP2',
            status='missing'
        )
        
        dataset_path = self.base_path / self.DATASET_EXPECTATIONS['pamap2']['path']
        
        if not dataset_path.exists():
            result.errors.append(f"Dataset directory does not exist: {dataset_path}")
            result.status = 'missing'
            return result
        
        extensions = self.DATASET_EXPECTATIONS['pamap2']['file_extensions']
        result.actual_count = self.count_files_in_directory(dataset_path, extensions, recursive=True)
        result.total_files = result.actual_count
        
        if result.actual_count == 0:
            result.errors.append("No data files found in dataset directory")
            result.status = 'missing'
            return result
        
        # Sample validation
        sample_files = list(dataset_path.rglob('*'))[:20]
        sample_files = [f for f in sample_files if f.is_file() and f.suffix.lower() in extensions]
        
        valid_count = 0
        for file_path in sample_files:
            is_valid, _ = self.validate_csv_file(file_path)
            if is_valid:
                valid_count += 1
                result.file_integrity_checks[str(file_path)] = True
            else:
                result.file_integrity_checks[str(file_path)] = False
        
        if len(sample_files) > 0:
            valid_ratio = valid_count / len(sample_files)
            result.valid_files = int(result.actual_count * valid_ratio)
            result.invalid_files = result.actual_count - result.valid_files
        else:
            result.valid_files = result.actual_count
            result.invalid_files = 0
        
        result.status = 'valid' if result.invalid_files == 0 and result.actual_count > 0 else 'partial'
        
        return result
    
    def validate_all(self) -> Dict[str, ValidationResult]:
        """
        Validate all datasets.
        
        Returns:
            Dictionary mapping dataset names to validation results
        """
        self.logger.info("Starting validation of all datasets...")
        
        self.results['esc50'] = self.validate_esc50()
        self.results['ravdess'] = self.validate_ravdess()
        self.results['audioset'] = self.validate_audioset()
        self.results['wesad'] = self.validate_wesad()
        self.results['pamap2'] = self.validate_pamap2()
        
        return self.results
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate validation report.
        
        Args:
            output_file: Optional path to save report JSON
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total_datasets = len(self.results)
        valid_count = sum(1 for r in self.results.values() if r.status == 'valid')
        partial_count = sum(1 for r in self.results.values() if r.status == 'partial')
        missing_count = sum(1 for r in self.results.values() if r.status == 'missing')
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Datasets: {total_datasets}")
        report_lines.append(f"✓ Valid: {valid_count}")
        report_lines.append(f"⚠ Partial: {partial_count}")
        report_lines.append(f"✗ Missing: {missing_count}")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 80)
        
        for dataset_name, result in self.results.items():
            report_lines.append(f"\n{result.dataset_name} ({dataset_name.upper()})")
            report_lines.append(f"  Status: {result.status.upper()}")
            report_lines.append(f"  Total Files: {result.total_files}")
            
            if result.expected_count:
                report_lines.append(f"  Expected: {result.expected_count}")
                report_lines.append(f"  Actual: {result.actual_count}")
                if result.expected_count > 0:
                    percentage = (result.actual_count / result.expected_count) * 100
                    report_lines.append(f"  Coverage: {percentage:.1f}%")
            
            report_lines.append(f"  Valid Files: {result.valid_files}")
            report_lines.append(f"  Invalid Files: {result.invalid_files}")
            
            if result.errors:
                report_lines.append(f"  Errors ({len(result.errors)}):")
                for error in result.errors[:5]:  # Show first 5 errors
                    report_lines.append(f"    - {error}")
                if len(result.errors) > 5:
                    report_lines.append(f"    ... and {len(result.errors) - 5} more errors")
            
            if result.warnings:
                report_lines.append(f"  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:3]:
                    report_lines.append(f"    - {warning}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to: {output_file}")
        
        # Also save JSON version
        if output_file:
            json_file = output_file.with_suffix('.json')
            results_dict = {name: asdict(result) for name, result in self.results.items()}
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, default=str)
            self.logger.info(f"JSON report saved to: {json_file}")
        
        return report


def main():
    """Main function to run dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate datasets for integrity, format, and counts')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['esc50', 'ravdess', 'audioset', 'wesad', 'pamap2', 'all'],
        default=['all'],
        help='Datasets to validate (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='logs/dataset_validation_report.txt',
        help='Output file for validation report (default: logs/dataset_validation_report.txt)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to project (default: script parent directory)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(__name__, log_file='dataset_validation.log', level=logging.INFO)
    
    base_path = Path(args.base_path) if args.base_path else project_root
    validator = DatasetValidator(base_path=base_path, logger=logger)
    
    # Validate specified datasets
    if 'all' in args.datasets:
        results = validator.validate_all()
    else:
        results = {}
        for dataset in args.datasets:
            if dataset == 'esc50':
                results['esc50'] = validator.validate_esc50()
            elif dataset == 'ravdess':
                results['ravdess'] = validator.validate_ravdess()
            elif dataset == 'audioset':
                results['audioset'] = validator.validate_audioset()
            elif dataset == 'wesad':
                results['wesad'] = validator.validate_wesad()
            elif dataset == 'pamap2':
                results['pamap2'] = validator.validate_pamap2()
        validator.results = results
    
    # Generate and print report
    output_path = Path(args.output)
    report = validator.generate_report(output_path)
    print(report)
    
    # Exit code based on results
    has_errors = any(r.status == 'missing' for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

