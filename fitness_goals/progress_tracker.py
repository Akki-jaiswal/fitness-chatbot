"""
Progress tracking functionality for fitness goals.

This module provides the ProgressTracker class that handles progress logging,
trend analysis, and visualization for fitness goal tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from statistics import mean, median
import math

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .models import FitnessGoal, ProgressEntry, validate_progress_data
from .data_storage import DataStorage, DataStorageError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressTrackerError(Exception):
    """Custom exception for progress tracking operations."""
    pass


class TrendAnalysis:
    """Data class for trend analysis results."""
    
    def __init__(self, status: str, progress_percentage: float, 
                 trend_direction: str, days_remaining: int,
                 current_rate: float, required_rate: float,
                 confidence: float, message: str):
        self.status = status  # 'on_track', 'ahead', 'behind'
        self.progress_percentage = progress_percentage
        self.trend_direction = trend_direction  # 'improving', 'declining', 'stable'
        self.days_remaining = days_remaining
        self.current_rate = current_rate  # kg/week
        self.required_rate = required_rate  # kg/week
        self.confidence = confidence  # 0.0 to 1.0
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trend analysis to dictionary."""
        return {
            'status': self.status,
            'progress_percentage': self.progress_percentage,
            'trend_direction': self.trend_direction,
            'days_remaining': self.days_remaining,
            'current_rate': self.current_rate,
            'required_rate': self.required_rate,
            'confidence': self.confidence,
            'message': self.message
        }


class ProgressTracker:
    """
    Manages progress tracking with logging, validation, and trend analysis.
    
    Provides methods to log progress entries, analyze trends, and generate
    insights for fitness goal tracking.
    """
    
    def __init__(self, data_storage: Optional[DataStorage] = None):
        """
        Initialize ProgressTracker with data storage backend.
        
        Args:
            data_storage: DataStorage instance (creates new if None)
        """
        self.data_storage = data_storage or DataStorage()
        logger.info("ProgressTracker initialized")
    
    def log_progress(self, user_id: str, progress_data: Dict[str, Any]) -> ProgressEntry:
        """
        Log a progress entry for the user with validation.
        
        Args:
            user_id: User identifier
            progress_data: Dictionary containing progress information
            
        Returns:
            Created ProgressEntry instance
            
        Raises:
            ProgressTrackerError: If validation fails or logging fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise ProgressTrackerError("Invalid user ID")
            
            # Validate progress data
            is_valid, error_msg = self.validate_progress_data(progress_data)
            if not is_valid:
                raise ProgressTrackerError(f"Progress validation failed: {error_msg}")
            
            # Prepare progress data with defaults
            processed_data = self._prepare_progress_data(progress_data)
            
            # Create progress entry object
            entry = ProgressEntry(**processed_data)
            
            # Additional validation against user's goal if available
            try:
                from .goal_manager import get_goal_manager
                goal_manager = get_goal_manager()
                user_goal = goal_manager.get_user_goal(user_id)
                
                if user_goal:
                    validation_result = self._validate_against_goal(entry, user_goal)
                    if not validation_result[0]:
                        logger.warning(f"Progress entry validation warning: {validation_result[1]}")
                        # Don't fail, just log warning for unrealistic values
            except Exception as e:
                logger.warning(f"Could not validate against goal: {str(e)}")
            
            # Save to storage
            success = self.data_storage.save_progress_entry(user_id, entry)
            if not success:
                raise ProgressTrackerError("Failed to save progress entry to storage")
            
            logger.info(f"Progress entry logged successfully for user {user_id}")
            return entry
            
        except DataStorageError as e:
            logger.error(f"Storage error logging progress for user {user_id}: {str(e)}")
            raise ProgressTrackerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error logging progress for user {user_id}: {str(e)}")
            raise ProgressTrackerError(f"Failed to log progress: {str(e)}")
    
    def get_progress_history(self, user_id: str, limit: Optional[int] = None) -> List[ProgressEntry]:
        """
        Retrieve progress history for the user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            List of ProgressEntry instances, sorted by date (most recent first)
            
        Raises:
            ProgressTrackerError: If retrieval fails
        """
        try:
            # Validate user ID
            if not user_id or not isinstance(user_id, str):
                raise ProgressTrackerError("Invalid user ID")
            
            history = self.data_storage.load_progress_history(user_id)
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                history = history[:limit]
            
            logger.info(f"Retrieved {len(history)} progress entries for user {user_id}")
            return history
            
        except DataStorageError as e:
            logger.error(f"Storage error retrieving progress for user {user_id}: {str(e)}")
            raise ProgressTrackerError(f"Storage error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving progress for user {user_id}: {str(e)}")
            raise ProgressTrackerError(f"Failed to retrieve progress history: {str(e)}")
    
    def calculate_trend(self, goal: FitnessGoal, history: List[ProgressEntry]) -> TrendAnalysis:
        """
        Calculate trend analysis based on goal and progress history.
        
        Args:
            goal: User's fitness goal
            history: List of progress entries (should be sorted by date)
            
        Returns:
            TrendAnalysis object with comprehensive trend information
            
        Raises:
            ProgressTrackerError: If analysis fails
        """
        try:
            if not isinstance(goal, FitnessGoal):
                raise ProgressTrackerError("Invalid goal object")
            
            if not history:
                return self._create_no_data_trend(goal)
            
            # Filter entries with weight data
            weight_entries = [entry for entry in history if entry.weight is not None]
            
            if len(weight_entries) < 2:
                return self._create_insufficient_data_trend(goal, weight_entries)
            
            # Sort by date (oldest first for trend calculation)
            weight_entries.sort(key=lambda x: x.date)
            
            # Calculate current progress
            latest_weight = weight_entries[-1].weight
            progress_percentage = goal.get_progress_percentage(latest_weight)
            
            # Calculate time progress
            days_remaining = goal.days_remaining()
            total_days = (goal.target_date - goal.created_date).days
            days_elapsed = total_days - days_remaining
            time_progress = (days_elapsed / total_days) if total_days > 0 else 0
            
            # Calculate current rate of change
            current_rate = self._calculate_rate_of_change(weight_entries)
            
            # Calculate required rate to meet goal
            required_rate = self._calculate_required_rate(goal, latest_weight, days_remaining)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(weight_entries)
            
            # Determine status (on track, ahead, behind)
            status = self._determine_status(
                goal, progress_percentage, time_progress, 
                current_rate, required_rate
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(weight_entries, days_elapsed)
            
            # Generate descriptive message
            message = self._generate_trend_message(
                goal, status, progress_percentage, current_rate, 
                required_rate, days_remaining, trend_direction
            )
            
            return TrendAnalysis(
                status=status,
                progress_percentage=progress_percentage,
                trend_direction=trend_direction,
                days_remaining=days_remaining,
                current_rate=current_rate,
                required_rate=required_rate,
                confidence=confidence,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend analysis: {str(e)}")
            raise ProgressTrackerError(f"Failed to calculate trend: {str(e)}")
    
    def generate_progress_chart(self, goal: FitnessGoal, history: List[ProgressEntry]) -> Optional[Any]:
        """
        Generate progress visualization chart with goal target line and trend indicators.
        
        Args:
            goal: User's fitness goal
            history: List of progress entries with weight data
            
        Returns:
            Plotly figure object or None if visualization cannot be created
            
        Raises:
            ProgressTrackerError: If chart generation fails
        """
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available for chart generation")
                return None
            
            # Filter entries with weight data and sort by date
            weight_entries = [entry for entry in history if entry.weight is not None]
            if not weight_entries:
                logger.info("No weight data available for chart generation")
                return None
            
            # Sort by date (oldest first)
            weight_entries.sort(key=lambda x: x.date)
            
            # Prepare data for plotting
            dates = [entry.date for entry in weight_entries]
            weights = [entry.weight for entry in weight_entries]
            
            # Create figure with secondary y-axis for trend indicators
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f"{goal.goal_type.replace('_', ' ').title()} Progress"],
                specs=[[{"secondary_y": False}]]
            )
            
            # Add weight progress line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=weights,
                    mode='lines+markers',
                    name='Weight Progress',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8, color='#2E86AB'),
                    hovertemplate='<b>%{x}</b><br>Weight: %{y:.1f} kg<extra></extra>'
                )
            )
            
            # Add goal target line
            target_line_dates = [goal.created_date, goal.target_date]
            target_line_weights = [goal.current_value, goal.target_value]
            
            fig.add_trace(
                go.Scatter(
                    x=target_line_dates,
                    y=target_line_weights,
                    mode='lines',
                    name='Goal Target',
                    line=dict(color='#A23B72', width=2, dash='dash'),
                    hovertemplate='<b>Target Line</b><br>%{x}: %{y:.1f} kg<extra></extra>'
                )
            )
            
            # Add milestone markers
            self._add_milestone_markers(fig, goal, weight_entries)
            
            # Add trend indicators
            self._add_trend_indicators(fig, goal, weight_entries)
            
            # Add current status annotation
            if weight_entries:
                latest_entry = weight_entries[-1]
                progress_pct = goal.get_progress_percentage(latest_entry.weight)
                
                fig.add_annotation(
                    x=latest_entry.date,
                    y=latest_entry.weight,
                    text=f"Current: {latest_entry.weight:.1f} kg<br>Progress: {progress_pct:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#2E86AB",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#2E86AB",
                    borderwidth=1
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Fitness Goal Progress - {goal.goal_type.replace('_', ' ').title()}",
                    x=0.5,
                    font=dict(size=20)
                ),
                xaxis_title="Date",
                yaxis_title="Weight (kg)",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.3)'
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.3)'
            )
            
            logger.info(f"Progress chart generated successfully with {len(weight_entries)} data points")
            return fig
            
        except Exception as e:
            logger.error(f"Error generating progress chart: {str(e)}")
            raise ProgressTrackerError(f"Failed to generate progress chart: {str(e)}")
    
    def _add_milestone_markers(self, fig, goal: FitnessGoal, weight_entries: List[ProgressEntry]) -> None:
        """
        Add milestone markers to the progress chart.
        
        Args:
            fig: Plotly figure object
            goal: User's fitness goal
            weight_entries: List of progress entries with weight data
        """
        try:
            if not weight_entries:
                return
            
            # Calculate milestone points (25%, 50%, 75%, 100% progress)
            milestones = [0.25, 0.5, 0.75, 1.0]
            milestone_colors = ['#F18F01', '#C73E1D', '#A23B72', '#2E8B57']
            
            for i, milestone in enumerate(milestones):
                # Calculate target weight for this milestone
                weight_change = goal.target_value - goal.current_value
                milestone_weight = goal.current_value + (weight_change * milestone)
                
                # Check if user has reached this milestone
                reached = False
                milestone_date = None
                
                for entry in weight_entries:
                    if goal.goal_type == 'weight_loss':
                        if entry.weight <= milestone_weight:
                            reached = True
                            milestone_date = entry.date
                            break
                    elif goal.goal_type == 'muscle_building':
                        if entry.weight >= milestone_weight:
                            reached = True
                            milestone_date = entry.date
                            break
                    else:  # weight_maintenance
                        tolerance = abs(weight_change) * 0.1  # 10% tolerance
                        if abs(entry.weight - milestone_weight) <= tolerance:
                            reached = True
                            milestone_date = entry.date
                            break
                
                if reached and milestone_date:
                    # Add milestone marker
                    fig.add_trace(
                        go.Scatter(
                            x=[milestone_date],
                            y=[milestone_weight],
                            mode='markers',
                            name=f'{int(milestone*100)}% Milestone',
                            marker=dict(
                                size=15,
                                color=milestone_colors[i],
                                symbol='star',
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate=f'<b>{int(milestone*100)}% Milestone Reached!</b><br>%{{x}}<br>Weight: %{{y:.1f}} kg<extra></extra>',
                            showlegend=False
                        )
                    )
                    
                    # Add milestone annotation
                    fig.add_annotation(
                        x=milestone_date,
                        y=milestone_weight,
                        text=f"🎉 {int(milestone*100)}%",
                        showarrow=False,
                        yshift=20,
                        font=dict(size=12, color=milestone_colors[i])
                    )
        
        except Exception as e:
            logger.warning(f"Error adding milestone markers: {str(e)}")
    
    def _add_trend_indicators(self, fig, goal: FitnessGoal, weight_entries: List[ProgressEntry]) -> None:
        """
        Add trend indicators to the progress chart.
        
        Args:
            fig: Plotly figure object
            goal: User's fitness goal
            weight_entries: List of progress entries with weight data
        """
        try:
            if len(weight_entries) < 3:
                return
            
            # Calculate trend analysis
            trend = self.calculate_trend(goal, weight_entries)
            
            # Add trend zone background
            dates = [entry.date for entry in weight_entries]
            weights = [entry.weight for entry in weight_entries]
            
            # Determine trend zone color based on status
            zone_colors = {
                'ahead': 'rgba(46, 139, 87, 0.1)',  # Green
                'on_track': 'rgba(255, 193, 7, 0.1)',  # Yellow
                'behind': 'rgba(220, 53, 69, 0.1)'  # Red
            }
            
            zone_color = zone_colors.get(trend.status, 'rgba(128, 128, 128, 0.1)')
            
            # Add trend zone
            fig.add_trace(
                go.Scatter(
                    x=dates + dates[::-1],
                    y=[min(weights) - 2] * len(dates) + [max(weights) + 2] * len(dates),
                    fill='toself',
                    fillcolor=zone_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Trend: {trend.status.replace("_", " ").title()}',
                    hoverinfo='skip',
                    showlegend=False
                )
            )
            
            # Add trend line (linear regression)
            if len(weight_entries) >= 2:
                trend_line_dates, trend_line_weights = self._calculate_trend_line(weight_entries)
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_line_dates,
                        y=trend_line_weights,
                        mode='lines',
                        name='Trend Line',
                        line=dict(
                            color='rgba(128, 128, 128, 0.8)',
                            width=2,
                            dash='dot'
                        ),
                        hovertemplate='<b>Trend Line</b><br>%{x}: %{y:.1f} kg<extra></extra>',
                        showlegend=False
                    )
                )
            
            # Add status indicator
            status_text = {
                'ahead': '🟢 Ahead of Schedule',
                'on_track': '🟡 On Track',
                'behind': '🔴 Behind Schedule'
            }
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=status_text.get(trend.status, '⚪ Status Unknown'),
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(128,128,128,0.5)",
                borderwidth=1,
                font=dict(size=14)
            )
            
        except Exception as e:
            logger.warning(f"Error adding trend indicators: {str(e)}")
    
    def _calculate_trend_line(self, weight_entries: List[ProgressEntry]) -> Tuple[List[datetime], List[float]]:
        """
        Calculate linear regression trend line for weight data.
        
        Args:
            weight_entries: List of progress entries with weight data
            
        Returns:
            Tuple of (dates, weights) for trend line
        """
        try:
            if len(weight_entries) < 2:
                return [], []
            
            # Convert dates to numeric values (days since first entry)
            first_date = weight_entries[0].date
            x_values = [(entry.date - first_date).days for entry in weight_entries]
            y_values = [entry.weight for entry in weight_entries]
            
            # Calculate linear regression
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return [], []
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Generate trend line points
            trend_dates = [weight_entries[0].date, weight_entries[-1].date]
            trend_weights = [
                intercept,  # Start point
                intercept + slope * x_values[-1]  # End point
            ]
            
            return trend_dates, trend_weights
            
        except Exception as e:
            logger.warning(f"Error calculating trend line: {str(e)}")
            return [], []

    def get_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive progress summary for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with progress summary information
            
        Raises:
            ProgressTrackerError: If summary generation fails
        """
        try:
            # Get user's goal
            from .goal_manager import get_goal_manager
            goal_manager = get_goal_manager()
            goal = goal_manager.get_user_goal(user_id)
            
            if not goal:
                return {
                    'has_goal': False,
                    'message': 'No active goal found'
                }
            
            # Get progress history
            history = self.get_progress_history(user_id)
            
            # Calculate trend analysis
            trend = self.calculate_trend(goal, history)
            
            # Calculate statistics
            stats = self._calculate_progress_statistics(history)
            
            # Get recent entries
            recent_entries = history[:5]  # Last 5 entries
            
            summary = {
                'has_goal': True,
                'goal_type': goal.goal_type,
                'goal_summary': {
                    'current_value': goal.current_value,
                    'target_value': goal.target_value,
                    'target_date': goal.target_date.isoformat(),
                    'days_remaining': goal.days_remaining(),
                    'is_overdue': goal.is_overdue()
                },
                'trend_analysis': trend.to_dict(),
                'statistics': stats,
                'recent_entries': [
                    {
                        'date': entry.date.isoformat(),
                        'weight': entry.weight,
                        'notes': entry.notes
                    }
                    for entry in recent_entries
                ],
                'total_entries': len(history),
                'last_updated': history[0].date.isoformat() if history else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating progress summary for user {user_id}: {str(e)}")
            raise ProgressTrackerError(f"Failed to generate progress summary: {str(e)}")
    
    def validate_progress_data(self, progress_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate progress data with enhanced business rules.
        
        Args:
            progress_data: Dictionary containing progress information
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use the model's validation first
            is_valid, error_msg = validate_progress_data(progress_data)
            if not is_valid:
                return False, error_msg
            
            # Additional business rule validations
            weight = progress_data.get('weight')
            body_measurements = progress_data.get('body_measurements', {})
            workout_achievements = progress_data.get('workout_achievements', {})
            
            # Validate weight change reasonableness (if we have previous data)
            # This would require access to previous entries, which we'll skip for now
            # to avoid circular dependencies
            
            # Validate body measurements consistency
            if body_measurements:
                # Check for reasonable measurement combinations
                waist = body_measurements.get('waist')
                chest = body_measurements.get('chest')
                
                if waist and chest and waist > chest * 1.5:
                    return False, "Waist measurement seems disproportionately large compared to chest"
            
            # Validate workout achievements consistency
            if workout_achievements:
                duration = workout_achievements.get('duration_minutes', 0)
                calories = workout_achievements.get('calories_burned', 0)
                
                if duration > 0 and calories > 0:
                    # Rough calorie burn rate validation (3-15 calories per minute is reasonable)
                    calorie_rate = calories / duration
                    if calorie_rate < 3 or calorie_rate > 15:
                        return False, (
                            f"Calorie burn rate of {calorie_rate:.1f} cal/min seems unrealistic. "
                            "Expected range: 3-15 cal/min"
                        )
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _prepare_progress_data(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare progress data with defaults and calculated fields.
        
        Args:
            progress_data: Raw progress data dictionary
            
        Returns:
            Processed progress data dictionary
        """
        processed_data = progress_data.copy()
        
        # Set default date if not provided
        if 'date' not in processed_data:
            processed_data['date'] = datetime.now()
        
        # Ensure date is datetime object
        if isinstance(processed_data['date'], str):
            processed_data['date'] = datetime.fromisoformat(processed_data['date'])
        
        return processed_data
    
    def _validate_against_goal(self, entry: ProgressEntry, goal: FitnessGoal) -> Tuple[bool, str]:
        """
        Validate progress entry against user's goal for reasonableness.
        
        Args:
            entry: Progress entry to validate
            goal: User's fitness goal
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if entry.weight is None:
            return True, ""
        
        # Check for extreme weight changes
        weight_diff = abs(entry.weight - goal.current_value)
        
        # Calculate days since goal creation
        days_since_start = (entry.date - goal.created_date).days
        
        if days_since_start > 0:
            # Maximum safe weight change rate (1 kg per week)
            max_safe_change = (days_since_start / 7) * 1.0
            
            if weight_diff > max_safe_change * 2:  # Allow some flexibility
                return False, (
                    f"Weight change of {weight_diff:.1f}kg in {days_since_start} days "
                    f"seems unusually large. Please verify the measurement."
                )
        
        # Check if weight is moving in wrong direction for the goal
        if goal.goal_type == 'weight_loss' and entry.weight > goal.current_value + 2:
            return False, (
                f"Weight has increased significantly from starting weight. "
                f"This may indicate measurement error or need for goal adjustment."
            )
        elif goal.goal_type == 'muscle_building' and entry.weight < goal.current_value - 2:
            return False, (
                f"Weight has decreased significantly from starting weight. "
                f"This may indicate measurement error or need for goal adjustment."
            )
        
        return True, ""
    
    def _calculate_rate_of_change(self, weight_entries: List[ProgressEntry]) -> float:
        """
        Calculate current rate of weight change in kg per week.
        
        Args:
            weight_entries: List of progress entries with weight data
            
        Returns:
            Rate of change in kg/week (positive for weight gain, negative for loss)
        """
        if len(weight_entries) < 2:
            return 0.0
        
        # Use recent entries for more accurate current rate
        recent_entries = weight_entries[-min(4, len(weight_entries)):]
        
        if len(recent_entries) < 2:
            return 0.0
        
        # Calculate linear regression slope
        weights = [entry.weight for entry in recent_entries]
        days = [(entry.date - recent_entries[0].date).days for entry in recent_entries]
        
        if len(set(days)) < 2:  # All entries on same day
            return 0.0
        
        # Simple linear regression
        n = len(weights)
        sum_x = sum(days)
        sum_y = sum(weights)
        sum_xy = sum(x * y for x, y in zip(days, weights))
        sum_x2 = sum(x * x for x in days)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Convert from kg/day to kg/week
        return slope * 7
    
    def _calculate_required_rate(self, goal: FitnessGoal, current_weight: float, 
                               days_remaining: int) -> float:
        """
        Calculate required rate of change to meet goal.
        
        Args:
            goal: User's fitness goal
            current_weight: Current weight from latest entry
            days_remaining: Days remaining to reach goal
            
        Returns:
            Required rate in kg/week
        """
        if days_remaining <= 0:
            return 0.0
        
        weight_change_needed = goal.target_value - current_weight
        weeks_remaining = days_remaining / 7
        
        return weight_change_needed / weeks_remaining if weeks_remaining > 0 else 0.0
    
    def _determine_trend_direction(self, weight_entries: List[ProgressEntry]) -> str:
        """
        Determine trend direction based on recent weight changes.
        
        Args:
            weight_entries: List of progress entries with weight data
            
        Returns:
            Trend direction: 'improving', 'declining', or 'stable'
        """
        if len(weight_entries) < 3:
            return 'stable'
        
        # Look at recent trend (last 3-5 entries)
        recent_entries = weight_entries[-min(5, len(weight_entries)):]
        weights = [entry.weight for entry in recent_entries]
        
        # Calculate trend using simple moving average comparison
        if len(weights) >= 3:
            early_avg = mean(weights[:len(weights)//2])
            late_avg = mean(weights[len(weights)//2:])
            
            diff = late_avg - early_avg
            
            if abs(diff) < 0.2:  # Less than 200g change
                return 'stable'
            elif diff > 0:
                return 'improving' if weights[-1] > weights[0] else 'declining'
            else:
                return 'improving' if weights[-1] < weights[0] else 'declining'
        
        return 'stable'
    
    def _determine_status(self, goal: FitnessGoal, progress_percentage: float, 
                         time_progress: float, current_rate: float, 
                         required_rate: float) -> str:
        """
        Determine if user is on track, ahead, or behind their goal.
        
        Args:
            goal: User's fitness goal
            progress_percentage: Current progress percentage
            time_progress: Time elapsed percentage
            current_rate: Current rate of change
            required_rate: Required rate of change
            
        Returns:
            Status: 'on_track', 'ahead', or 'behind'
        """
        # Compare progress vs time
        progress_vs_time = progress_percentage - (time_progress * 100)
        
        # Compare current rate vs required rate
        rate_comparison = abs(current_rate) - abs(required_rate)
        
        # Determine status based on multiple factors
        if progress_vs_time > 10 or rate_comparison > 0.2:
            return 'ahead'
        elif progress_vs_time < -10 or rate_comparison < -0.2:
            return 'behind'
        else:
            return 'on_track'
    
    def _calculate_confidence(self, weight_entries: List[ProgressEntry], 
                            days_elapsed: int) -> float:
        """
        Calculate confidence level of trend analysis.
        
        Args:
            weight_entries: List of progress entries with weight data
            days_elapsed: Days since goal started
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Base confidence on number of data points
        data_points = len(weight_entries)
        if data_points >= 5:
            confidence += 0.4
        elif data_points >= 3:
            confidence += 0.3
        elif data_points >= 2:
            confidence += 0.2
        
        # Increase confidence with time elapsed
        if days_elapsed >= 14:
            confidence += 0.3
        elif days_elapsed >= 7:
            confidence += 0.2
        elif days_elapsed >= 3:
            confidence += 0.1
        
        # Increase confidence with data consistency
        if data_points >= 3:
            weights = [entry.weight for entry in weight_entries]
            weight_variance = sum((w - mean(weights))**2 for w in weights) / len(weights)
            
            if weight_variance < 1.0:  # Low variance indicates consistent measurements
                confidence += 0.3
            elif weight_variance < 4.0:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _generate_trend_message(self, goal: FitnessGoal, status: str, 
                              progress_percentage: float, current_rate: float,
                              required_rate: float, days_remaining: int,
                              trend_direction: str) -> str:
        """
        Generate descriptive message about trend analysis.
        
        Args:
            goal: User's fitness goal
            status: Current status (on_track, ahead, behind)
            progress_percentage: Current progress percentage
            current_rate: Current rate of change
            required_rate: Required rate of change
            days_remaining: Days remaining to goal
            trend_direction: Trend direction
            
        Returns:
            Descriptive message string
        """
        goal_type_text = {
            'weight_loss': 'weight loss',
            'muscle_building': 'muscle building',
            'weight_maintenance': 'weight maintenance'
        }
        
        goal_text = goal_type_text.get(goal.goal_type, 'fitness')
        
        if status == 'ahead':
            return (
                f"Great progress! You're ahead of schedule on your {goal_text} goal. "
                f"You've achieved {progress_percentage:.1f}% of your target with "
                f"{days_remaining} days remaining. Keep up the excellent work!"
            )
        elif status == 'behind':
            rate_diff = abs(required_rate) - abs(current_rate)
            return (
                f"You're currently behind on your {goal_text} goal. "
                f"You've achieved {progress_percentage:.1f}% of your target. "
                f"Consider increasing your effort by {rate_diff:.2f} kg/week to get back on track."
            )
        else:  # on_track
            return (
                f"You're on track with your {goal_text} goal! "
                f"You've achieved {progress_percentage:.1f}% of your target. "
                f"Continue your current approach to reach your goal in {days_remaining} days."
            )
    
    def _create_no_data_trend(self, goal: FitnessGoal) -> TrendAnalysis:
        """Create trend analysis when no progress data is available."""
        return TrendAnalysis(
            status='on_track',
            progress_percentage=0.0,
            trend_direction='stable',
            days_remaining=goal.days_remaining(),
            current_rate=0.0,
            required_rate=self._calculate_required_rate(goal, goal.current_value, goal.days_remaining()),
            confidence=0.0,
            message="No progress data available yet. Start logging your progress to see trend analysis."
        )
    
    def _create_insufficient_data_trend(self, goal: FitnessGoal, 
                                      entries: List[ProgressEntry]) -> TrendAnalysis:
        """Create trend analysis when insufficient data is available."""
        latest_weight = entries[0].weight if entries else goal.current_value
        progress_percentage = goal.get_progress_percentage(latest_weight)
        
        return TrendAnalysis(
            status='on_track',
            progress_percentage=progress_percentage,
            trend_direction='stable',
            days_remaining=goal.days_remaining(),
            current_rate=0.0,
            required_rate=self._calculate_required_rate(goal, latest_weight, goal.days_remaining()),
            confidence=0.1,
            message="More progress data needed for accurate trend analysis. Keep logging your progress!"
        )
    
    def _calculate_progress_statistics(self, history: List[ProgressEntry]) -> Dict[str, Any]:
        """
        Calculate statistical information about progress history.
        
        Args:
            history: List of progress entries
            
        Returns:
            Dictionary with statistical information
        """
        stats = {
            'total_entries': len(history),
            'entries_with_weight': 0,
            'entries_with_measurements': 0,
            'entries_with_workouts': 0,
            'date_range_days': 0,
            'average_weight': None,
            'weight_change_total': None,
            'most_recent_weight': None
        }
        
        if not history:
            return stats
        
        # Count entry types
        weight_entries = []
        for entry in history:
            if entry.weight is not None:
                stats['entries_with_weight'] += 1
                weight_entries.append(entry.weight)
            if entry.body_measurements:
                stats['entries_with_measurements'] += 1
            if entry.workout_achievements:
                stats['entries_with_workouts'] += 1
        
        # Calculate date range
        if len(history) > 1:
            sorted_history = sorted(history, key=lambda x: x.date)
            date_range = sorted_history[-1].date - sorted_history[0].date
            stats['date_range_days'] = date_range.days
        
        # Calculate weight statistics
        if weight_entries:
            stats['average_weight'] = mean(weight_entries)
            stats['most_recent_weight'] = weight_entries[0]  # history is sorted newest first
            if len(weight_entries) > 1:
                stats['weight_change_total'] = weight_entries[0] - weight_entries[-1]
        
        return stats


# Global instance for easy access
_progress_tracker_instance = None


def get_progress_tracker() -> ProgressTracker:
    """
    Get the global ProgressTracker instance (singleton pattern).
    
    Returns:
        ProgressTracker instance
    """
    global _progress_tracker_instance
    if _progress_tracker_instance is None:
        _progress_tracker_instance = ProgressTracker()
    return _progress_tracker_instance