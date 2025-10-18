"""
Utility functions for fitness goals module.

This module provides helper functions for data processing,
validation, and common operations.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .models import FitnessGoal, ProgressEntry


def create_goal_from_user_input(
    goal_type: str,
    current_value: float,
    target_value: float,
    timeframe_weeks: int
) -> FitnessGoal:
    """
    Create a FitnessGoal from user input with automatic date calculation.
    
    Args:
        goal_type: Type of fitness goal
        current_value: Current measurement value
        target_value: Target measurement value
        timeframe_weeks: Duration in weeks
        
    Returns:
        FitnessGoal instance
        
    Raises:
        ValueError: If input data is invalid
    """
    created_date = datetime.now()
    target_date = created_date + timedelta(weeks=timeframe_weeks)
    
    return FitnessGoal(
        goal_type=goal_type,
        current_value=current_value,
        target_value=target_value,
        timeframe_weeks=timeframe_weeks,
        created_date=created_date,
        target_date=target_date
    )


def create_progress_entry_from_user_input(
    weight: Optional[float] = None,
    measurements: Optional[Dict[str, float]] = None,
    workout_data: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None
) -> ProgressEntry:
    """
    Create a ProgressEntry from user input with current timestamp.
    
    Args:
        weight: Weight measurement
        measurements: Body measurements dictionary
        workout_data: Workout achievements dictionary
        notes: User notes
        
    Returns:
        ProgressEntry instance
        
    Raises:
        ValueError: If no data is provided or data is invalid
    """
    return ProgressEntry(
        date=datetime.now(),
        weight=weight,
        body_measurements=measurements,
        workout_achievements=workout_data,
        notes=notes
    )


def calculate_weekly_progress_needed(goal: FitnessGoal, current_value: float) -> float:
    """
    Calculate the weekly progress needed to achieve the goal on time.
    
    Args:
        goal: The fitness goal
        current_value: Current progress value
        
    Returns:
        Weekly progress needed (positive for increase, negative for decrease)
    """
    weeks_remaining = goal.days_remaining() / 7
    if weeks_remaining <= 0:
        return 0.0
    
    total_change_needed = goal.target_value - current_value
    return total_change_needed / weeks_remaining


def get_progress_trend(
    goal: FitnessGoal,
    progress_history: List[ProgressEntry],
    lookback_days: int = 14
) -> Dict[str, Any]:
    """
    Analyze progress trend over recent entries.
    
    Args:
        goal: The fitness goal
        progress_history: List of progress entries
        lookback_days: Number of days to look back for trend analysis
        
    Returns:
        Dictionary with trend analysis results
    """
    if not progress_history:
        return {
            'trend': 'no_data',
            'message': 'No progress data available',
            'recent_entries': 0
        }
    
    # Filter recent entries with weight data
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent_entries = [
        entry for entry in progress_history
        if entry.date >= cutoff_date and entry.weight is not None
    ]
    
    if len(recent_entries) < 2:
        return {
            'trend': 'insufficient_data',
            'message': f'Need at least 2 weight entries in the last {lookback_days} days',
            'recent_entries': len(recent_entries)
        }
    
    # Sort by date
    recent_entries.sort(key=lambda x: x.date)
    
    # Calculate trend
    first_weight = recent_entries[0].weight
    last_weight = recent_entries[-1].weight
    weight_change = last_weight - first_weight
    
    # Determine if trend aligns with goal
    if goal.goal_type == 'weight_loss':
        if weight_change < -0.5:  # Losing weight
            trend = 'on_track'
            message = f'Great! You\'ve lost {abs(weight_change):.1f} kg in the last {lookback_days} days'
        elif weight_change > 0.5:  # Gaining weight
            trend = 'off_track'
            message = f'You\'ve gained {weight_change:.1f} kg. Consider adjusting your approach'
        else:
            trend = 'stable'
            message = 'Weight is stable. You might need to adjust your calorie deficit'
    
    elif goal.goal_type == 'muscle_building':
        if weight_change > 0.2:  # Gaining weight
            trend = 'on_track'
            message = f'Good progress! You\'ve gained {weight_change:.1f} kg in the last {lookback_days} days'
        elif weight_change < -0.2:  # Losing weight
            trend = 'off_track'
            message = f'You\'ve lost {abs(weight_change):.1f} kg. Consider increasing calorie intake'
        else:
            trend = 'stable'
            message = 'Weight is stable. This is normal for muscle building'
    
    else:  # weight_maintenance
        if abs(weight_change) <= 1.0:  # Within 1kg
            trend = 'on_track'
            message = f'Excellent! Weight maintained within {abs(weight_change):.1f} kg'
        else:
            trend = 'off_track'
            message = f'Weight changed by {weight_change:.1f} kg. Consider adjusting your approach'
    
    return {
        'trend': trend,
        'message': message,
        'recent_entries': len(recent_entries),
        'weight_change': weight_change,
        'days_analyzed': lookback_days
    }


def get_goal_status_summary(goal: FitnessGoal, latest_progress: Optional[ProgressEntry]) -> Dict[str, Any]:
    """
    Get a comprehensive status summary for a goal.
    
    Args:
        goal: The fitness goal
        latest_progress: Most recent progress entry (if any)
        
    Returns:
        Dictionary with goal status information
    """
    days_remaining = goal.days_remaining()
    is_overdue = goal.is_overdue()
    
    if latest_progress and latest_progress.weight is not None:
        current_value = latest_progress.weight
        progress_percentage = goal.get_progress_percentage(current_value)
        weekly_needed = calculate_weekly_progress_needed(goal, current_value)
    else:
        current_value = goal.current_value
        progress_percentage = 0.0
        weekly_needed = calculate_weekly_progress_needed(goal, current_value)
    
    # Determine status
    if is_overdue:
        status = 'overdue'
        status_message = 'Goal deadline has passed'
    elif progress_percentage >= 100:
        status = 'completed'
        status_message = 'Congratulations! Goal achieved!'
    elif days_remaining <= 7:
        status = 'urgent'
        status_message = f'Only {days_remaining} days left!'
    elif progress_percentage >= 75:
        status = 'on_track'
        status_message = 'Great progress! You\'re almost there'
    elif progress_percentage >= 25:
        status = 'moderate'
        status_message = 'Making progress, keep it up!'
    else:
        status = 'behind'
        status_message = 'Consider adjusting your approach'
    
    return {
        'status': status,
        'status_message': status_message,
        'progress_percentage': progress_percentage,
        'days_remaining': days_remaining,
        'weeks_remaining': days_remaining / 7,
        'current_value': current_value,
        'target_value': goal.target_value,
        'weekly_progress_needed': weekly_needed,
        'is_overdue': is_overdue,
        'goal_type': goal.goal_type
    }


def format_goal_summary(goal: FitnessGoal) -> str:
    """
    Format a human-readable goal summary.
    
    Args:
        goal: The fitness goal
        
    Returns:
        Formatted string describing the goal
    """
    goal_type_names = {
        'weight_loss': 'Weight Loss',
        'muscle_building': 'Muscle Building',
        'weight_maintenance': 'Weight Maintenance'
    }
    
    goal_name = goal_type_names.get(goal.goal_type, goal.goal_type.title())
    
    return (
        f"{goal_name} Goal: "
        f"{goal.current_value:.1f} kg → {goal.target_value:.1f} kg "
        f"in {goal.timeframe_weeks} weeks "
        f"(by {goal.target_date.strftime('%B %d, %Y')})"
    )


def format_progress_summary(entry: ProgressEntry) -> str:
    """
    Format a human-readable progress entry summary.
    
    Args:
        entry: The progress entry
        
    Returns:
        Formatted string describing the progress
    """
    parts = []
    
    if entry.weight is not None:
        parts.append(f"Weight: {entry.weight:.1f} kg")
    
    if entry.body_measurements:
        measurements = ", ".join([
            f"{key}: {value:.1f} cm"
            for key, value in entry.body_measurements.items()
        ])
        parts.append(f"Measurements: {measurements}")
    
    if entry.workout_achievements:
        workout_parts = []
        if 'duration_minutes' in entry.workout_achievements:
            workout_parts.append(f"{entry.workout_achievements['duration_minutes']} min")
        if 'calories_burned' in entry.workout_achievements:
            workout_parts.append(f"{entry.workout_achievements['calories_burned']} cal")
        if workout_parts:
            parts.append(f"Workout: {', '.join(workout_parts)}")
    
    if entry.notes:
        parts.append(f"Notes: {entry.notes}")
    
    date_str = entry.date.strftime('%B %d, %Y')
    summary = " | ".join(parts) if parts else "No data recorded"
    
    return f"{date_str}: {summary}"