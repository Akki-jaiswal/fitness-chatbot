"""
Goal management functionality for fitness goals.

This module provides the GoalManager class that handles CRUD operations
for fitness goals with validation and business logic.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import uuid

from .models import FitnessGoal, validate_goal_data
from .data_storage import DataStorage, DataStorageError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalManagerError(Exception):
    """Custom exception for goal management operations."""
    pass


class GoalManager:
    """
    Manages fitness goal CRUD operations with validation and business logic.
    
    Provides methods to create, read, update, and delete fitness goals
    while ensuring data integrity and business rule compliance.
    """
    
    def __init__(self, data_storage: Optional[DataStorage] = None):
        """
        Initialize GoalManager with data storage backend.
        
        Args:
            data_storage: DataStorage instance (creates new if None)
        """
        self.data_storage = data_storage or DataStorage()
        logger.info("GoalManager initialized")
    
    def create_goal(self, user_id: str, goal_data: Dict[str, Any]) -> FitnessGoal:
        """
        Create a new fitness goal for the user.
        
        Args:
            user_id: User identifier
            goal_data: Dictionary containing goal information
            
        Returns:
            Created FitnessGoal instance
            
        Raises:
            GoalManagerError: If validation fails or goal creation fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            # Check if user already has an active goal
            existing_goal = self.get_user_goal(user_id)
            if existing_goal and existing_goal.is_active:
                raise GoalManagerError(
                    f"User already has an active {existing_goal.goal_type} goal. "
                    "Please update or deactivate the existing goal first."
                )
            
            # Validate goal data
            is_valid, error_msg = self.validate_goal_data(goal_data)
            if not is_valid:
                raise GoalManagerError(f"Goal validation failed: {error_msg}")
            
            # Prepare goal data with defaults
            processed_data = self._prepare_goal_data(goal_data)
            
            # Create goal object
            goal = FitnessGoal(**processed_data)
            
            # Save to storage
            success = self.data_storage.save_goal(user_id, goal)
            if not success:
                raise GoalManagerError("Failed to save goal to storage")
            
            logger.info(f"Goal created successfully for user {user_id}: {goal.goal_type}")
            return goal
            
        except DataStorageError as e:
            logger.error(f"Storage error creating goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to create goal: {str(e)}")
    
    def get_user_goal(self, user_id: str) -> Optional[FitnessGoal]:
        """
        Retrieve the current fitness goal for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            FitnessGoal instance if found, None otherwise
            
        Raises:
            GoalManagerError: If retrieval fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            goal = self.data_storage.load_goal(user_id)
            
            if goal:
                logger.info(f"Goal retrieved for user {user_id}: {goal.goal_type}")
            else:
                logger.info(f"No goal found for user {user_id}")
            
            return goal
            
        except DataStorageError as e:
            logger.error(f"Storage error retrieving goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to retrieve goal: {str(e)}")
    
    def update_goal(self, user_id: str, updates: Dict[str, Any]) -> FitnessGoal:
        """
        Update an existing fitness goal for the user.
        
        Args:
            user_id: User identifier
            updates: Dictionary containing fields to update
            
        Returns:
            Updated FitnessGoal instance
            
        Raises:
            GoalManagerError: If validation fails or update fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            # Get existing goal
            existing_goal = self.get_user_goal(user_id)
            if not existing_goal:
                raise GoalManagerError("No existing goal found to update")
            
            # Prepare updated goal data
            updated_data = existing_goal.to_dict()
            
            # Apply updates with validation
            allowed_updates = {
                'target_value', 'timeframe_weeks', 'is_active'
            }
            
            for key, value in updates.items():
                if key not in allowed_updates:
                    raise GoalManagerError(f"Field '{key}' cannot be updated")
                updated_data[key] = value
            
            # Recalculate target date if timeframe changed
            if 'timeframe_weeks' in updates:
                updated_data['target_date'] = (
                    existing_goal.created_date + 
                    timedelta(weeks=updates['timeframe_weeks'])
                )
            
            # Validate updated goal data
            is_valid, error_msg = self.validate_goal_data(updated_data)
            if not is_valid:
                raise GoalManagerError(f"Updated goal validation failed: {error_msg}")
            
            # Create updated goal object
            updated_goal = FitnessGoal.from_dict(updated_data)
            
            # Save to storage
            success = self.data_storage.save_goal(user_id, updated_goal)
            if not success:
                raise GoalManagerError("Failed to save updated goal to storage")
            
            logger.info(f"Goal updated successfully for user {user_id}")
            return updated_goal
            
        except GoalManagerError:
            raise
        except DataStorageError as e:
            logger.error(f"Storage error updating goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error updating goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to update goal: {str(e)}")
    
    def replace_goal(self, user_id: str, new_goal_data: Dict[str, Any]) -> FitnessGoal:
        """
        Replace the existing goal with a completely new goal.
        
        Args:
            user_id: User identifier
            new_goal_data: Dictionary containing new goal information
            
        Returns:
            New FitnessGoal instance
            
        Raises:
            GoalManagerError: If validation fails or replacement fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            # Deactivate existing goal if it exists
            existing_goal = self.get_user_goal(user_id)
            if existing_goal:
                logger.info(f"Replacing existing {existing_goal.goal_type} goal for user {user_id}")
            
            # Create new goal (this will validate and save)
            new_goal = self.create_goal(user_id, new_goal_data)
            
            logger.info(f"Goal replaced successfully for user {user_id}: {new_goal.goal_type}")
            return new_goal
            
        except GoalManagerError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error replacing goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to replace goal: {str(e)}")
    
    def deactivate_goal(self, user_id: str) -> bool:
        """
        Deactivate the current goal for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False if no goal exists
            
        Raises:
            GoalManagerError: If deactivation fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            # Get existing goal
            existing_goal = self.get_user_goal(user_id)
            if not existing_goal:
                logger.info(f"No goal to deactivate for user {user_id}")
                return False
            
            # Update goal to inactive
            updated_goal = self.update_goal(user_id, {'is_active': False})
            
            logger.info(f"Goal deactivated for user {user_id}")
            return True
            
        except GoalManagerError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error deactivating goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to deactivate goal: {str(e)}")
    
    def delete_goal(self, user_id: str) -> bool:
        """
        Delete the goal and all associated data for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False if no goal exists
            
        Raises:
            GoalManagerError: If deletion fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise GoalManagerError("Invalid user ID")
            
            # Check if goal exists
            existing_goal = self.get_user_goal(user_id)
            if not existing_goal:
                logger.info(f"No goal to delete for user {user_id}")
                return False
            
            # Clear user data (includes goal and progress)
            success = self.data_storage.clear_user_data(user_id)
            if not success:
                raise GoalManagerError("Failed to delete goal from storage")
            
            logger.info(f"Goal deleted for user {user_id}")
            return True
            
        except DataStorageError as e:
            logger.error(f"Storage error deleting goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error deleting goal for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to delete goal: {str(e)}")
    
    def validate_goal_data(self, goal_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate goal data with enhanced business rules.
        
        Args:
            goal_data: Dictionary containing goal information
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use the model's validation first
            is_valid, error_msg = validate_goal_data(goal_data)
            if not is_valid:
                return False, error_msg
            
            # Additional business rule validations
            goal_type = goal_data.get('goal_type')
            current_value = goal_data.get('current_value')
            target_value = goal_data.get('target_value')
            timeframe_weeks = goal_data.get('timeframe_weeks')
            
            # Validate realistic weight change rates
            if goal_type in ['weight_loss', 'muscle_building']:
                weight_change = abs(target_value - current_value)
                max_safe_rate = 1.0  # kg per week (safe weight change rate)
                max_change = max_safe_rate * timeframe_weeks
                
                if weight_change > max_change:
                    return False, (
                        f"Target weight change of {weight_change:.1f}kg in {timeframe_weeks} weeks "
                        f"exceeds safe rate of {max_safe_rate}kg/week. "
                        f"Consider extending timeframe or adjusting target."
                    )
            
            # Validate minimum meaningful change
            if goal_type in ['weight_loss', 'muscle_building']:
                min_change = 2.0  # Minimum 2kg change to be meaningful
                weight_change = abs(target_value - current_value)
                
                if weight_change < min_change:
                    return False, (
                        f"Target weight change of {weight_change:.1f}kg is too small. "
                        f"Consider a minimum change of {min_change}kg for meaningful results."
                    )
            
            # Validate maintenance goal tolerance
            if goal_type == 'weight_maintenance':
                tolerance = abs(target_value - current_value)
                max_tolerance = 3.0  # Maximum 3kg difference for maintenance
                
                if tolerance > max_tolerance:
                    return False, (
                        f"Weight maintenance target differs by {tolerance:.1f}kg from current weight. "
                        f"For maintenance goals, target should be within {max_tolerance}kg of current weight."
                    )
            
            # Validate timeframe appropriateness
            if goal_type == 'weight_loss':
                # Minimum 4 weeks for any weight loss goal
                if timeframe_weeks < 4:
                    return False, "Weight loss goals require at least 4 weeks"
                
                # Recommend longer timeframes for larger changes
                weight_change = current_value - target_value
                if weight_change > 10 and timeframe_weeks < 12:
                    return False, (
                        f"Weight loss of {weight_change:.1f}kg should take at least 12 weeks "
                        "for sustainable results"
                    )
            
            elif goal_type == 'muscle_building':
                # Muscle building takes time
                if timeframe_weeks < 8:
                    return False, "Muscle building goals require at least 8 weeks"
                
                weight_gain = target_value - current_value
                if weight_gain > 5 and timeframe_weeks < 16:
                    return False, (
                        f"Muscle gain of {weight_gain:.1f}kg should take at least 16 weeks "
                        "for quality results"
                    )
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _prepare_goal_data(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare goal data with defaults and calculated fields.
        
        Args:
            goal_data: Raw goal data dictionary
            
        Returns:
            Processed goal data dictionary
        """
        processed_data = goal_data.copy()
        
        # Set default dates if not provided
        if 'created_date' not in processed_data:
            processed_data['created_date'] = datetime.now()
        
        if 'target_date' not in processed_data:
            created_date = processed_data['created_date']
            if isinstance(created_date, str):
                created_date = datetime.fromisoformat(created_date)
            
            timeframe_weeks = processed_data['timeframe_weeks']
            processed_data['target_date'] = created_date + timedelta(weeks=timeframe_weeks)
        
        # Set default values
        if 'goal_id' not in processed_data:
            processed_data['goal_id'] = str(uuid.uuid4())
        
        if 'is_active' not in processed_data:
            processed_data['is_active'] = True
        
        return processed_data
    
    def get_goal_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the user's current goal with progress information.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with goal summary or None if no goal exists
            
        Raises:
            GoalManagerError: If retrieval fails
        """
        try:
            goal = self.get_user_goal(user_id)
            if not goal:
                return None
            
            # Calculate time-based information
            days_remaining = goal.days_remaining()
            is_overdue = goal.is_overdue()
            total_days = (goal.target_date - goal.created_date).days
            days_elapsed = total_days - days_remaining
            
            # Calculate progress percentage (without current progress data)
            # This would typically use actual progress data from ProgressTracker
            time_progress = (days_elapsed / total_days * 100) if total_days > 0 else 0
            
            summary = {
                'goal_id': goal.goal_id,
                'goal_type': goal.goal_type,
                'current_value': goal.current_value,
                'target_value': goal.target_value,
                'timeframe_weeks': goal.timeframe_weeks,
                'created_date': goal.created_date.isoformat(),
                'target_date': goal.target_date.isoformat(),
                'is_active': goal.is_active,
                'days_remaining': days_remaining,
                'days_elapsed': days_elapsed,
                'total_days': total_days,
                'is_overdue': is_overdue,
                'time_progress_percentage': min(100, max(0, time_progress)),
                'target_change': abs(goal.target_value - goal.current_value),
                'change_direction': 'decrease' if goal.target_value < goal.current_value else 'increase'
            }
            
            return summary
            
        except GoalManagerError:
            raise
        except Exception as e:
            logger.error(f"Error getting goal summary for user {user_id}: {str(e)}")
            raise GoalManagerError(f"Failed to get goal summary: {str(e)}")


# Global instance for easy access
_goal_manager_instance = None


def get_goal_manager() -> GoalManager:
    """
    Get the global GoalManager instance (singleton pattern).
    
    Returns:
        GoalManager instance
    """
    global _goal_manager_instance
    if _goal_manager_instance is None:
        _goal_manager_instance = GoalManager()
    return _goal_manager_instance