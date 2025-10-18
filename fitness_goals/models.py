"""
Core data models for fitness goals and progress tracking.

This module defines the FitnessGoal and ProgressEntry data classes
with validation and serialization capabilities.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import uuid


@dataclass
class FitnessGoal:
    """
    Represents a user's fitness goal with validation and serialization.
    
    Attributes:
        goal_type: Type of goal ('weight_loss', 'muscle_building', 'weight_maintenance')
        current_value: Current measurement value (e.g., weight in kg)
        target_value: Target measurement value
        timeframe_weeks: Duration in weeks (4-52 weeks)
        created_date: When the goal was created
        target_date: When the goal should be achieved
        goal_id: Unique identifier for the goal
        is_active: Whether the goal is currently active
    """
    goal_type: str
    current_value: float
    target_value: float
    timeframe_weeks: int
    created_date: datetime
    target_date: datetime
    goal_id: str = None
    is_active: bool = True
    
    def __post_init__(self):
        """Validate goal data after initialization."""
        if self.goal_id is None:
            self.goal_id = str(uuid.uuid4())
        
        self._validate_goal_type()
        self._validate_values()
        self._validate_timeframe()
        self._validate_dates()
    
    def _validate_goal_type(self):
        """Validate that goal_type is one of the allowed values."""
        valid_types = ['weight_loss', 'muscle_building', 'weight_maintenance']
        if self.goal_type not in valid_types:
            raise ValueError(f"Goal type must be one of: {valid_types}")
    
    def _validate_values(self):
        """Validate current and target values are reasonable."""
        if self.current_value <= 0:
            raise ValueError("Current value must be positive")
        if self.target_value <= 0:
            raise ValueError("Target value must be positive")
        
        # Validate reasonable weight ranges (assuming kg)
        if self.current_value < 30 or self.current_value > 300:
            raise ValueError("Current value seems unrealistic (expected 30-300 kg)")
        if self.target_value < 30 or self.target_value > 300:
            raise ValueError("Target value seems unrealistic (expected 30-300 kg)")
        
        # Validate goal makes sense for goal type
        if self.goal_type == 'weight_loss' and self.target_value >= self.current_value:
            raise ValueError("Weight loss goal requires target value less than current value")
        elif self.goal_type == 'muscle_building' and self.target_value <= self.current_value:
            raise ValueError("Muscle building goal requires target value greater than current value")
    
    def _validate_timeframe(self):
        """Validate timeframe is within reasonable bounds."""
        if not isinstance(self.timeframe_weeks, int):
            raise ValueError("Timeframe must be an integer")
        if self.timeframe_weeks < 4 or self.timeframe_weeks > 52:
            raise ValueError("Timeframe must be between 4 and 52 weeks")
    
    def _validate_dates(self):
        """Validate dates are reasonable."""
        if self.created_date > datetime.now():
            raise ValueError("Created date cannot be in the future")
        if self.target_date <= self.created_date:
            raise ValueError("Target date must be after created date")
        
        # Check if target date matches timeframe
        expected_target = self.created_date + timedelta(weeks=self.timeframe_weeks)
        if abs((self.target_date - expected_target).days) > 1:
            raise ValueError("Target date doesn't match specified timeframe")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['created_date'] = self.created_date.isoformat()
        data['target_date'] = self.target_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FitnessGoal':
        """Create FitnessGoal from dictionary."""
        # Convert ISO format strings back to datetime objects
        if isinstance(data['created_date'], str):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if isinstance(data['target_date'], str):
            data['target_date'] = datetime.fromisoformat(data['target_date'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert goal to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FitnessGoal':
        """Create FitnessGoal from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_progress_percentage(self, current_progress: float) -> float:
        """Calculate progress percentage towards goal."""
        if self.goal_type == 'weight_maintenance':
            # For maintenance, calculate how close to target (smaller difference = higher percentage)
            max_acceptable_diff = abs(self.current_value - self.target_value) or 1
            current_diff = abs(current_progress - self.target_value)
            return max(0, (1 - current_diff / max_acceptable_diff) * 100)
        else:
            # For weight loss/gain, calculate linear progress
            total_change_needed = abs(self.target_value - self.current_value)
            change_achieved = abs(current_progress - self.current_value)
            return min(100, (change_achieved / total_change_needed) * 100)
    
    def days_remaining(self) -> int:
        """Get number of days remaining to achieve goal."""
        return max(0, (self.target_date - datetime.now()).days)
    
    def is_overdue(self) -> bool:
        """Check if goal deadline has passed."""
        return datetime.now() > self.target_date


@dataclass
class ProgressEntry:
    """
    Represents a single progress tracking entry.
    
    Attributes:
        date: When the progress was recorded
        weight: Weight measurement in kg
        body_measurements: Dictionary of body measurements
        workout_achievements: Dictionary of workout data
        notes: Optional user notes
        entry_id: Unique identifier for the entry
    """
    date: datetime
    weight: Optional[float] = None
    body_measurements: Optional[Dict[str, float]] = None
    workout_achievements: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    entry_id: str = None
    
    def __post_init__(self):
        """Validate progress entry data after initialization."""
        if self.entry_id is None:
            self.entry_id = str(uuid.uuid4())
        
        self._validate_date()
        self._validate_weight()
        self._validate_measurements()
        self._validate_workout_achievements()
    
    def _validate_date(self):
        """Validate entry date is reasonable."""
        if self.date > datetime.now():
            raise ValueError("Progress entry date cannot be in the future")
        
        # Don't allow entries older than 2 years
        two_years_ago = datetime.now() - timedelta(days=730)
        if self.date < two_years_ago:
            raise ValueError("Progress entry date cannot be more than 2 years old")
    
    def _validate_weight(self):
        """Validate weight measurement if provided."""
        if self.weight is not None:
            if self.weight <= 0:
                raise ValueError("Weight must be positive")
            if self.weight < 30 or self.weight > 300:
                raise ValueError("Weight seems unrealistic (expected 30-300 kg)")
    
    def _validate_measurements(self):
        """Validate body measurements if provided."""
        if self.body_measurements is not None:
            if not isinstance(self.body_measurements, dict):
                raise ValueError("Body measurements must be a dictionary")
            
            valid_measurements = ['waist', 'chest', 'arms', 'thighs', 'hips', 'neck']
            for key, value in self.body_measurements.items():
                if key not in valid_measurements:
                    raise ValueError(f"Invalid measurement type: {key}")
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Measurement {key} must be a positive number")
                if value > 200:  # Reasonable upper bound in cm
                    raise ValueError(f"Measurement {key} seems unrealistic")
    
    def _validate_workout_achievements(self):
        """Validate workout achievements if provided."""
        if self.workout_achievements is not None:
            if not isinstance(self.workout_achievements, dict):
                raise ValueError("Workout achievements must be a dictionary")
            
            # Validate specific workout fields if present
            if 'duration_minutes' in self.workout_achievements:
                duration = self.workout_achievements['duration_minutes']
                if not isinstance(duration, (int, float)) or duration <= 0:
                    raise ValueError("Workout duration must be positive")
                if duration > 480:  # 8 hours max
                    raise ValueError("Workout duration seems unrealistic")
            
            if 'calories_burned' in self.workout_achievements:
                calories = self.workout_achievements['calories_burned']
                if not isinstance(calories, (int, float)) or calories <= 0:
                    raise ValueError("Calories burned must be positive")
                if calories > 2000:  # Reasonable upper bound
                    raise ValueError("Calories burned seems unrealistic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress entry to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data['date'] = self.date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressEntry':
        """Create ProgressEntry from dictionary."""
        # Convert ISO format string back to datetime
        if isinstance(data['date'], str):
            data['date'] = datetime.fromisoformat(data['date'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert progress entry to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProgressEntry':
        """Create ProgressEntry from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def has_data(self) -> bool:
        """Check if entry contains any actual data."""
        return (self.weight is not None or 
                (self.body_measurements and len(self.body_measurements) > 0) or
                (self.workout_achievements and len(self.workout_achievements) > 0) or
                (self.notes and len(self.notes.strip()) > 0))


def validate_goal_data(goal_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate goal data dictionary before creating FitnessGoal.
    
    Args:
        goal_data: Dictionary containing goal information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields
        required_fields = ['goal_type', 'current_value', 'target_value', 'timeframe_weeks']
        for field in required_fields:
            if field not in goal_data:
                return False, f"Missing required field: {field}"
        
        # Make a copy to avoid modifying the original
        data_copy = goal_data.copy()
        
        # Convert string dates to datetime objects if needed
        if 'created_date' in data_copy and isinstance(data_copy['created_date'], str):
            data_copy['created_date'] = datetime.fromisoformat(data_copy['created_date'])
        if 'target_date' in data_copy and isinstance(data_copy['target_date'], str):
            data_copy['target_date'] = datetime.fromisoformat(data_copy['target_date'])
        
        # Add dates if not provided
        if 'created_date' not in data_copy:
            data_copy['created_date'] = datetime.now()
        if 'target_date' not in data_copy:
            data_copy['target_date'] = data_copy['created_date'] + timedelta(weeks=data_copy['timeframe_weeks'])
        
        # Try to create goal (this will validate all constraints)
        FitnessGoal(**data_copy)
        return True, ""
        
    except (ValueError, TypeError) as e:
        return False, str(e)


def validate_progress_data(progress_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate progress data dictionary before creating ProgressEntry.
    
    Args:
        progress_data: Dictionary containing progress information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Make a copy to avoid modifying the original
        data_copy = progress_data.copy()
        
        # Convert string date to datetime object if needed
        if 'date' in data_copy and isinstance(data_copy['date'], str):
            data_copy['date'] = datetime.fromisoformat(data_copy['date'])
        
        # Add current date if not provided
        if 'date' not in data_copy:
            data_copy['date'] = datetime.now()
        
        # Try to create progress entry (this will validate all constraints)
        entry = ProgressEntry(**data_copy)
        
        # Check that entry has some actual data
        if not entry.has_data():
            return False, "Progress entry must contain at least one measurement or note"
        
        return True, ""
        
    except (ValueError, TypeError) as e:
        return False, str(e)