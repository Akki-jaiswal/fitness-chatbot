"""
Data persistence layer for fitness goals and progress tracking.

This module provides the DataStorage class that handles data persistence
using Streamlit session state and browser local storage integration.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
import base64
import io
import zipfile

from .models import FitnessGoal, ProgressEntry, validate_goal_data, validate_progress_data


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStorageError(Exception):
    """Custom exception for data storage operations."""
    pass


class DataStorage:
    """
    Handles data persistence for fitness goals and progress tracking.
    
    Uses Streamlit session state as primary storage with browser local storage
    as backup for data persistence across sessions.
    """
    
    def __init__(self):
        """Initialize DataStorage with session state setup."""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values."""
        if 'fitness_goals' not in st.session_state:
            st.session_state.fitness_goals = {}
        
        if 'progress_history' not in st.session_state:
            st.session_state.progress_history = {}
        
        if 'user_id' not in st.session_state:
            # Generate a simple session-based user ID
            st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'data_storage_initialized' not in st.session_state:
            st.session_state.data_storage_initialized = True
            logger.info(f"DataStorage initialized for user: {st.session_state.user_id}")
    
    def _get_user_id(self) -> str:
        """Get the current user ID from session state."""
        return st.session_state.get('user_id', 'default_user')
    
    def _validate_user_id(self, user_id: str) -> bool:
        """Validate user ID format and content."""
        if not user_id or not isinstance(user_id, str):
            return False
        if len(user_id) > 100:  # Reasonable length limit
            return False
        return True
    
    def save_goal(self, user_id: str, goal: FitnessGoal) -> bool:
        """
        Save a fitness goal for the specified user.
        
        Args:
            user_id: User identifier
            goal: FitnessGoal instance to save
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DataStorageError: If validation fails or storage operation fails
        """
        try:
            # Validate inputs
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            if not isinstance(goal, FitnessGoal):
                raise DataStorageError("Goal must be a FitnessGoal instance")
            
            # Convert goal to dictionary for storage
            goal_data = goal.to_dict()
            
            # Validate goal data
            is_valid, error_msg = validate_goal_data(goal_data.copy())
            if not is_valid:
                raise DataStorageError(f"Goal validation failed: {error_msg}")
            
            # Store in session state
            if 'fitness_goals' not in st.session_state:
                st.session_state.fitness_goals = {}
            
            st.session_state.fitness_goals[user_id] = goal_data
            
            # Try to persist to browser local storage via JavaScript
            self._persist_to_browser_storage('fitness_goals', st.session_state.fitness_goals)
            
            logger.info(f"Goal saved successfully for user {user_id}: {goal.goal_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save goal for user {user_id}: {str(e)}")
            raise DataStorageError(f"Failed to save goal: {str(e)}")
    
    def load_goal(self, user_id: str) -> Optional[FitnessGoal]:
        """
        Load a fitness goal for the specified user.
        
        Args:
            user_id: User identifier
            
        Returns:
            FitnessGoal instance if found, None otherwise
            
        Raises:
            DataStorageError: If data is corrupted or invalid
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            # Try to load from session state first
            goals = st.session_state.get('fitness_goals', {})
            goal_data = goals.get(user_id)
            
            if goal_data is None:
                # Try to restore from browser storage
                self._restore_from_browser_storage()
                goals = st.session_state.get('fitness_goals', {})
                goal_data = goals.get(user_id)
            
            if goal_data is None:
                logger.info(f"No goal found for user {user_id}")
                return None
            
            # Validate and create goal object
            is_valid, error_msg = validate_goal_data(goal_data.copy())
            if not is_valid:
                logger.error(f"Invalid goal data for user {user_id}: {error_msg}")
                raise DataStorageError(f"Corrupted goal data: {error_msg}")
            
            goal = FitnessGoal.from_dict(goal_data)
            logger.info(f"Goal loaded successfully for user {user_id}: {goal.goal_type}")
            return goal
            
        except DataStorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to load goal for user {user_id}: {str(e)}")
            raise DataStorageError(f"Failed to load goal: {str(e)}")
    
    def save_progress_entry(self, user_id: str, entry: ProgressEntry) -> bool:
        """
        Save a progress entry for the specified user.
        
        Args:
            user_id: User identifier
            entry: ProgressEntry instance to save
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DataStorageError: If validation fails or storage operation fails
        """
        try:
            # Validate inputs
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            if not isinstance(entry, ProgressEntry):
                raise DataStorageError("Entry must be a ProgressEntry instance")
            
            # Convert entry to dictionary for storage
            entry_data = entry.to_dict()
            
            # Validate progress data
            is_valid, error_msg = validate_progress_data(entry_data.copy())
            if not is_valid:
                raise DataStorageError(f"Progress entry validation failed: {error_msg}")
            
            # Initialize progress history if needed
            if 'progress_history' not in st.session_state:
                st.session_state.progress_history = {}
            
            if user_id not in st.session_state.progress_history:
                st.session_state.progress_history[user_id] = []
            
            # Add entry to history (append with timestamp handling)
            user_history = st.session_state.progress_history[user_id]
            
            # Check for duplicate entries (same date and similar data)
            existing_entry = self._find_duplicate_entry(user_history, entry_data)
            if existing_entry:
                logger.warning(f"Duplicate progress entry detected for user {user_id}")
                # Update existing entry instead of creating duplicate
                existing_entry.update(entry_data)
            else:
                user_history.append(entry_data)
            
            # Sort history by date (most recent first)
            # Handle mixed datetime objects and ISO strings
            def get_sort_key(entry):
                date_val = entry['date']
                if isinstance(date_val, datetime):
                    return date_val
                else:
                    return datetime.fromisoformat(date_val)
            
            user_history.sort(key=get_sort_key, reverse=True)
            
            # Limit history size to prevent excessive memory usage (keep last 365 entries)
            if len(user_history) > 365:
                st.session_state.progress_history[user_id] = user_history[:365]
                logger.info(f"Trimmed progress history for user {user_id} to 365 entries")
            
            # Try to persist to browser local storage
            self._persist_to_browser_storage('progress_history', st.session_state.progress_history)
            
            logger.info(f"Progress entry saved successfully for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress entry for user {user_id}: {str(e)}")
            raise DataStorageError(f"Failed to save progress entry: {str(e)}")
    
    def load_progress_history(self, user_id: str) -> List[ProgressEntry]:
        """
        Load progress history for the specified user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of ProgressEntry instances, sorted by date (most recent first)
            
        Raises:
            DataStorageError: If data is corrupted or invalid
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            # Try to load from session state first
            history_data = st.session_state.get('progress_history', {})
            user_history = history_data.get(user_id, [])
            
            if not user_history:
                # Try to restore from browser storage
                self._restore_from_browser_storage()
                history_data = st.session_state.get('progress_history', {})
                user_history = history_data.get(user_id, [])
            
            if not user_history:
                logger.info(f"No progress history found for user {user_id}")
                return []
            
            # Convert to ProgressEntry objects and validate
            progress_entries = []
            invalid_entries = []
            
            for i, entry_data in enumerate(user_history):
                try:
                    # Validate entry data
                    is_valid, error_msg = validate_progress_data(entry_data.copy())
                    if not is_valid:
                        logger.warning(f"Invalid progress entry {i} for user {user_id}: {error_msg}")
                        invalid_entries.append(i)
                        continue
                    
                    entry = ProgressEntry.from_dict(entry_data)
                    progress_entries.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse progress entry {i} for user {user_id}: {str(e)}")
                    invalid_entries.append(i)
            
            # Remove invalid entries from session state
            if invalid_entries:
                logger.info(f"Removing {len(invalid_entries)} invalid entries for user {user_id}")
                cleaned_history = [
                    entry for i, entry in enumerate(user_history)
                    if i not in invalid_entries
                ]
                st.session_state.progress_history[user_id] = cleaned_history
                self._persist_to_browser_storage('progress_history', st.session_state.progress_history)
            
            logger.info(f"Loaded {len(progress_entries)} progress entries for user {user_id}")
            return progress_entries
            
        except DataStorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to load progress history for user {user_id}: {str(e)}")
            raise DataStorageError(f"Failed to load progress history: {str(e)}")
    
    def clear_user_data(self, user_id: str) -> bool:
        """
        Clear all data for the specified user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DataStorageError: If operation fails
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            # Clear from session state
            if 'fitness_goals' in st.session_state and user_id in st.session_state.fitness_goals:
                del st.session_state.fitness_goals[user_id]
            
            if 'progress_history' in st.session_state and user_id in st.session_state.progress_history:
                del st.session_state.progress_history[user_id]
            
            # Update browser storage
            self._persist_to_browser_storage('fitness_goals', st.session_state.get('fitness_goals', {}))
            self._persist_to_browser_storage('progress_history', st.session_state.get('progress_history', {}))
            
            logger.info(f"Cleared all data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear data for user {user_id}: {str(e)}")
            raise DataStorageError(f"Failed to clear user data: {str(e)}")
    
    def _find_duplicate_entry(self, history: List[Dict], new_entry: Dict) -> Optional[Dict]:
        """
        Find duplicate progress entry in history.
        
        Args:
            history: List of existing progress entries
            new_entry: New entry to check for duplicates
            
        Returns:
            Existing entry if duplicate found, None otherwise
        """
        new_date = new_entry['date']
        
        # Convert new_date to string if it's a datetime object
        if isinstance(new_date, datetime):
            new_date_str = new_date.isoformat()[:10]  # YYYY-MM-DD part
        else:
            new_date_str = new_date[:10]  # Assume it's already a string
        
        for entry in history:
            entry_date = entry['date']
            
            # Convert entry_date to string if it's a datetime object
            if isinstance(entry_date, datetime):
                entry_date_str = entry_date.isoformat()[:10]  # YYYY-MM-DD part
            else:
                entry_date_str = entry_date[:10]  # Assume it's already a string
            
            # Check if entries are from the same day
            if entry_date_str == new_date_str:
                # Consider it a duplicate if it's the same day
                return entry
        
        return None
    
    def _persist_to_browser_storage(self, key: str, data: Dict) -> None:
        """
        Persist data to browser local storage using Streamlit components.
        
        Args:
            key: Storage key
            data: Data to persist
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, default=str)
            
            # Use Streamlit's session state to trigger browser storage
            # This is a simplified approach - in a real implementation,
            # you might use custom Streamlit components or JavaScript
            storage_key = f"fitness_app_{key}"
            st.session_state[storage_key] = json_data
            
            logger.debug(f"Data persisted to browser storage: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to persist data to browser storage: {str(e)}")
            # Don't raise exception - browser storage is optional
    
    def _restore_from_browser_storage(self) -> None:
        """
        Restore data from browser local storage.
        """
        try:
            # Try to restore fitness goals
            goals_key = "fitness_app_fitness_goals"
            if goals_key in st.session_state:
                goals_data = json.loads(st.session_state[goals_key])
                st.session_state.fitness_goals = goals_data
                logger.debug("Restored fitness goals from browser storage")
            
            # Try to restore progress history
            history_key = "fitness_app_progress_history"
            if history_key in st.session_state:
                history_data = json.loads(st.session_state[history_key])
                st.session_state.progress_history = history_data
                logger.debug("Restored progress history from browser storage")
                
        except Exception as e:
            logger.warning(f"Failed to restore data from browser storage: {str(e)}")
            # Don't raise exception - browser storage restoration is optional
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            goals_count = len(st.session_state.get('fitness_goals', {}))
            
            progress_history = st.session_state.get('progress_history', {})
            total_entries = sum(len(history) for history in progress_history.values())
            users_with_progress = len(progress_history)
            
            return {
                'total_goals': goals_count,
                'total_progress_entries': total_entries,
                'users_with_progress': users_with_progress,
                'current_user_id': self._get_user_id(),
                'session_initialized': st.session_state.get('data_storage_initialized', False)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {
                'error': str(e),
                'total_goals': 0,
                'total_progress_entries': 0,
                'users_with_progress': 0,
                'current_user_id': 'unknown',
                'session_initialized': False
            }
    
    def validate_data_integrity(self, user_id: str) -> Dict[str, Any]:
        """
        Validate data integrity for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'user_id': user_id,
            'goal_valid': False,
            'goal_error': None,
            'progress_entries_valid': 0,
            'progress_entries_invalid': 0,
            'progress_errors': []
        }
        
        try:
            # Validate goal
            goal = self.load_goal(user_id)
            if goal:
                results['goal_valid'] = True
            
        except Exception as e:
            results['goal_error'] = str(e)
        
        try:
            # Validate progress history
            progress_history = self.load_progress_history(user_id)
            results['progress_entries_valid'] = len(progress_history)
            
        except Exception as e:
            results['progress_errors'].append(str(e))
        
        return results
    
    def clear_user_data_with_confirmation(self, user_id: str) -> Tuple[bool, str]:
        """
        Clear user data with confirmation dialog and backup option.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                return False, "Invalid user ID"
            
            # Check if user has any data to clear
            has_goal = user_id in st.session_state.get('fitness_goals', {})
            has_progress = user_id in st.session_state.get('progress_history', {})
            
            if not has_goal and not has_progress:
                return True, "No data found to clear"
            
            # Create backup before clearing
            backup_data = self.export_user_data(user_id)
            if backup_data:
                # Store backup in session state temporarily
                st.session_state['last_data_backup'] = {
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'data': backup_data
                }
            
            # Clear the data
            success = self.clear_user_data(user_id)
            
            if success:
                message = "All user data cleared successfully."
                if backup_data:
                    message += " A backup has been created and can be restored if needed."
                return True, message
            else:
                return False, "Failed to clear user data"
                
        except Exception as e:
            logger.error(f"Failed to clear user data with confirmation: {str(e)}")
            return False, f"Error clearing data: {str(e)}"
    
    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Export all user data for backup purposes.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing all user data, or None if no data found
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                raise DataStorageError("Invalid user ID")
            
            export_data = {
                'export_version': '1.0',
                'export_timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'goal': None,
                'progress_history': []
            }
            
            # Export goal data
            try:
                goal = self.load_goal(user_id)
                if goal:
                    export_data['goal'] = goal.to_dict()
            except Exception as e:
                logger.warning(f"Failed to export goal for user {user_id}: {str(e)}")
            
            # Export progress history
            try:
                progress_history = self.load_progress_history(user_id)
                export_data['progress_history'] = [entry.to_dict() for entry in progress_history]
            except Exception as e:
                logger.warning(f"Failed to export progress history for user {user_id}: {str(e)}")
            
            # Only return data if there's something to export
            if export_data['goal'] or export_data['progress_history']:
                logger.info(f"Exported data for user {user_id}: {len(export_data['progress_history'])} progress entries")
                return export_data
            else:
                logger.info(f"No data to export for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export user data: {str(e)}")
            return None
    
    def import_user_data(self, user_id: str, import_data: Dict[str, Any], overwrite: bool = False) -> Tuple[bool, str]:
        """
        Import user data from backup.
        
        Args:
            user_id: User identifier
            import_data: Data to import
            overwrite: Whether to overwrite existing data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate user ID
            if not self._validate_user_id(user_id):
                return False, "Invalid user ID"
            
            # Validate import data structure
            if not isinstance(import_data, dict):
                return False, "Invalid import data format"
            
            required_fields = ['export_version', 'export_timestamp', 'user_id']
            for field in required_fields:
                if field not in import_data:
                    return False, f"Missing required field in import data: {field}"
            
            # Check if user already has data and overwrite is not allowed
            if not overwrite:
                existing_goal = self.load_goal(user_id)
                existing_progress = self.load_progress_history(user_id)
                if existing_goal or existing_progress:
                    return False, "User already has data. Use overwrite=True to replace existing data."
            
            imported_items = []
            errors = []
            
            # Import goal data
            if 'goal' in import_data and import_data['goal']:
                try:
                    goal_data = import_data['goal']
                    # Validate goal data
                    is_valid, error_msg = validate_goal_data(goal_data)
                    if not is_valid:
                        errors.append(f"Invalid goal data: {error_msg}")
                    else:
                        goal = FitnessGoal.from_dict(goal_data)
                        success = self.save_goal(user_id, goal)
                        if success:
                            imported_items.append("goal")
                        else:
                            errors.append("Failed to save imported goal")
                except Exception as e:
                    errors.append(f"Error importing goal: {str(e)}")
            
            # Import progress history
            if 'progress_history' in import_data and import_data['progress_history']:
                try:
                    progress_data = import_data['progress_history']
                    imported_entries = 0
                    
                    for entry_data in progress_data:
                        try:
                            # Validate progress entry data
                            is_valid, error_msg = validate_progress_data(entry_data)
                            if not is_valid:
                                errors.append(f"Invalid progress entry: {error_msg}")
                                continue
                            
                            entry = ProgressEntry.from_dict(entry_data)
                            success = self.save_progress_entry(user_id, entry)
                            if success:
                                imported_entries += 1
                            else:
                                errors.append("Failed to save progress entry")
                        except Exception as e:
                            errors.append(f"Error importing progress entry: {str(e)}")
                    
                    if imported_entries > 0:
                        imported_items.append(f"{imported_entries} progress entries")
                        
                except Exception as e:
                    errors.append(f"Error importing progress history: {str(e)}")
            
            # Generate result message
            if imported_items:
                message = f"Successfully imported: {', '.join(imported_items)}"
                if errors:
                    message += f". Errors encountered: {'; '.join(errors[:3])}"  # Limit error messages
                logger.info(f"Data import completed for user {user_id}: {message}")
                return True, message
            else:
                error_msg = f"No data imported. Errors: {'; '.join(errors[:3])}" if errors else "No valid data found to import"
                return False, error_msg
                
        except Exception as e:
            logger.error(f"Failed to import user data: {str(e)}")
            return False, f"Import failed: {str(e)}"
    
    def create_backup_file(self, user_id: str) -> Optional[bytes]:
        """
        Create a downloadable backup file for user data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Backup file as bytes, or None if no data to backup
        """
        try:
            # Export user data
            export_data = self.export_user_data(user_id)
            if not export_data:
                return None
            
            # Create a ZIP file containing the backup
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add main data file
                data_json = json.dumps(export_data, indent=2, default=str)
                zip_file.writestr('fitness_data.json', data_json)
                
                # Add metadata file
                metadata = {
                    'backup_created': datetime.now().isoformat(),
                    'user_id': user_id,
                    'app_version': '1.0',
                    'data_format_version': export_data.get('export_version', '1.0')
                }
                metadata_json = json.dumps(metadata, indent=2)
                zip_file.writestr('backup_info.json', metadata_json)
                
                # Add readme file
                readme_content = """Fitness Goals Data Backup
========================

This backup contains your fitness goals and progress tracking data.

Files included:
- fitness_data.json: Your goals and progress entries
- backup_info.json: Backup metadata
- README.txt: This file

To restore this data:
1. Open the fitness goals application
2. Go to the data management section
3. Use the "Import Data" feature
4. Upload the fitness_data.json file

Created: {timestamp}
User ID: {user_id}
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id=user_id)
                
                zip_file.writestr('README.txt', readme_content)
            
            zip_buffer.seek(0)
            backup_bytes = zip_buffer.getvalue()
            
            logger.info(f"Created backup file for user {user_id}, size: {len(backup_bytes)} bytes")
            return backup_bytes
            
        except Exception as e:
            logger.error(f"Failed to create backup file: {str(e)}")
            return None
    
    def restore_from_backup_file(self, user_id: str, backup_file_bytes: bytes, overwrite: bool = False) -> Tuple[bool, str]:
        """
        Restore user data from a backup file.
        
        Args:
            user_id: User identifier
            backup_file_bytes: Backup file content as bytes
            overwrite: Whether to overwrite existing data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read the ZIP file
            zip_buffer = io.BytesIO(backup_file_bytes)
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                # Check if required files exist
                file_list = zip_file.namelist()
                if 'fitness_data.json' not in file_list:
                    return False, "Invalid backup file: missing fitness_data.json"
                
                # Read the main data file
                data_content = zip_file.read('fitness_data.json').decode('utf-8')
                import_data = json.loads(data_content)
                
                # Import the data
                return self.import_user_data(user_id, import_data, overwrite)
                
        except zipfile.BadZipFile:
            return False, "Invalid backup file: not a valid ZIP file"
        except json.JSONDecodeError:
            return False, "Invalid backup file: corrupted data format"
        except Exception as e:
            logger.error(f"Failed to restore from backup file: {str(e)}")
            return False, f"Restore failed: {str(e)}"
    
    def recover_corrupted_data(self, user_id: str) -> Dict[str, Any]:
        """
        Attempt to recover corrupted or missing data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with recovery results and actions taken
        """
        recovery_results = {
            'user_id': user_id,
            'actions_taken': [],
            'recovered_items': [],
            'unrecoverable_items': [],
            'warnings': []
        }
        
        try:
            # Check for backup in session state
            if 'last_data_backup' in st.session_state:
                backup_info = st.session_state['last_data_backup']
                if backup_info['user_id'] == user_id:
                    recovery_results['actions_taken'].append("Found recent backup in session")
                    
                    # Attempt to restore from backup
                    success, message = self.import_user_data(user_id, backup_info['data'], overwrite=True)
                    if success:
                        recovery_results['recovered_items'].append("Data restored from session backup")
                    else:
                        recovery_results['warnings'].append(f"Failed to restore from backup: {message}")
            
            # Try to restore from browser storage
            try:
                self._restore_from_browser_storage()
                recovery_results['actions_taken'].append("Attempted browser storage restoration")
            except Exception as e:
                recovery_results['warnings'].append(f"Browser storage restoration failed: {str(e)}")
            
            # Validate current data integrity
            integrity_results = self.validate_data_integrity(user_id)
            
            # Clean up invalid progress entries
            if integrity_results['progress_errors']:
                try:
                    # Reload progress history (this will clean invalid entries)
                    progress_history = self.load_progress_history(user_id)
                    recovery_results['actions_taken'].append("Cleaned invalid progress entries")
                    recovery_results['recovered_items'].append(f"Validated {len(progress_history)} progress entries")
                except Exception as e:
                    recovery_results['unrecoverable_items'].append(f"Progress history: {str(e)}")
            
            # Handle corrupted goal data
            if integrity_results['goal_error']:
                recovery_results['unrecoverable_items'].append(f"Goal data: {integrity_results['goal_error']}")
                
                # Try to clear corrupted goal data
                try:
                    if 'fitness_goals' in st.session_state and user_id in st.session_state.fitness_goals:
                        del st.session_state.fitness_goals[user_id]
                        recovery_results['actions_taken'].append("Removed corrupted goal data")
                except Exception:
                    pass
            
            # Initialize clean session state if needed
            if not integrity_results['goal_valid'] and not integrity_results['progress_entries_valid']:
                self._initialize_session_state()
                recovery_results['actions_taken'].append("Reinitialized session state")
            
            logger.info(f"Data recovery completed for user {user_id}: {len(recovery_results['actions_taken'])} actions taken")
            return recovery_results
            
        except Exception as e:
            logger.error(f"Data recovery failed for user {user_id}: {str(e)}")
            recovery_results['unrecoverable_items'].append(f"Recovery process failed: {str(e)}")
            return recovery_results
    
    def restore_session_from_browser_storage(self) -> Tuple[bool, str]:
        """
        Enhanced session restoration from browser local storage with error handling.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            restored_items = []
            errors = []
            
            # Try to restore fitness goals
            goals_key = "fitness_app_fitness_goals"
            if goals_key in st.session_state:
                try:
                    goals_data = json.loads(st.session_state[goals_key])
                    
                    # Validate each goal before restoring
                    valid_goals = {}
                    for user_id, goal_data in goals_data.items():
                        try:
                            is_valid, error_msg = validate_goal_data(goal_data)
                            if is_valid:
                                valid_goals[user_id] = goal_data
                            else:
                                errors.append(f"Invalid goal for user {user_id}: {error_msg}")
                        except Exception as e:
                            errors.append(f"Error validating goal for user {user_id}: {str(e)}")
                    
                    if valid_goals:
                        st.session_state.fitness_goals = valid_goals
                        restored_items.append(f"{len(valid_goals)} fitness goals")
                        
                except json.JSONDecodeError:
                    errors.append("Corrupted fitness goals data in browser storage")
                except Exception as e:
                    errors.append(f"Error restoring fitness goals: {str(e)}")
            
            # Try to restore progress history
            history_key = "fitness_app_progress_history"
            if history_key in st.session_state:
                try:
                    history_data = json.loads(st.session_state[history_key])
                    
                    # Validate each progress entry before restoring
                    valid_history = {}
                    total_entries = 0
                    
                    for user_id, user_history in history_data.items():
                        valid_entries = []
                        for entry_data in user_history:
                            try:
                                is_valid, error_msg = validate_progress_data(entry_data)
                                if is_valid:
                                    valid_entries.append(entry_data)
                                    total_entries += 1
                                else:
                                    errors.append(f"Invalid progress entry for user {user_id}: {error_msg}")
                            except Exception as e:
                                errors.append(f"Error validating progress entry for user {user_id}: {str(e)}")
                        
                        if valid_entries:
                            valid_history[user_id] = valid_entries
                    
                    if valid_history:
                        st.session_state.progress_history = valid_history
                        restored_items.append(f"{total_entries} progress entries")
                        
                except json.JSONDecodeError:
                    errors.append("Corrupted progress history data in browser storage")
                except Exception as e:
                    errors.append(f"Error restoring progress history: {str(e)}")
            
            # Generate result message
            if restored_items:
                message = f"Successfully restored: {', '.join(restored_items)}"
                if errors:
                    message += f". Some data could not be restored: {len(errors)} errors"
                logger.info(f"Session restoration completed: {message}")
                return True, message
            else:
                if errors:
                    message = f"No data restored due to errors: {'; '.join(errors[:3])}"
                else:
                    message = "No data found in browser storage to restore"
                return False, message
                
        except Exception as e:
            logger.error(f"Session restoration failed: {str(e)}")
            return False, f"Session restoration failed: {str(e)}"


# Global instance for easy access
_storage_instance = None


def get_data_storage() -> DataStorage:
    """
    Get the global DataStorage instance (singleton pattern).
    
    Returns:
        DataStorage instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DataStorage()
    return _storage_instance