"""
Progress tracking user interface components for Streamlit.

This module provides the UI components for logging progress entries,
displaying progress history, and showing progress summary statistics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from .progress_tracker import get_progress_tracker, ProgressTrackerError
from .goal_manager import get_goal_manager, GoalManagerError
from .models import ProgressEntry
from .data_storage import get_data_storage
from .ui_feedback import (
    show_success, show_error, show_warning, show_info, show_validation_error,
    loading_indicator, confirm_action, safe_operation, handle_ui_errors,
    validate_weight_input, validate_measurement_input
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressUIError(Exception):
    """Custom exception for progress UI operations."""
    pass


@handle_ui_errors("render progress tracking interface")
def render_progress_tracking_interface() -> None:
    """
    Render the complete progress tracking interface.
    
    This function creates all the UI components for progress tracking including:
    - Progress entry form with weight and measurement inputs
    - Progress history display with tabular format
    - Basic progress summary statistics display
    """
    with safe_operation("load progress tracking interface", 
                       loading_message="Loading progress tracking..."):
        # Initialize components
        progress_tracker = get_progress_tracker()
        goal_manager = get_goal_manager()
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Check if user has a goal with enhanced error handling
        user_goal = None
        try:
            with loading_indicator("Loading your goal...", "load_user_goal"):
                user_goal = goal_manager.get_user_goal(user_id)
        except GoalManagerError as e:
            show_error("Failed to load your goal", e, recoverable=True)
        except Exception as e:
            show_error("Unexpected error loading goal", e, recoverable=True)
        
        if not user_goal:
            show_info("Set a fitness goal first to start tracking your progress!", 
                     "You can set a goal using the sidebar on the left.")
            return
        
        # Check visualization availability with error handling
        show_visualization = False
        try:
            with loading_indicator("Checking data availability...", "check_viz"):
                history = progress_tracker.get_progress_history(user_id)
                weight_entries = [entry for entry in history if entry.weight is not None]
                show_visualization = len(weight_entries) >= 2
                
                if len(weight_entries) == 1:
                    show_info("Add one more weight entry to unlock progress visualization!")
                elif len(weight_entries) == 0:
                    show_info("Log your first weight entry to start tracking progress!")
        except Exception as e:
            logger.warning(f"Could not check visualization availability: {str(e)}")
            show_warning("Unable to check visualization availability", 
                        "Progress tracking will still work normally.")
        
        # Create tabs based on data availability
        if show_visualization:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Log Progress", 
                "📈 Progress History", 
                "📋 Summary", 
                "📊 Visualization"
            ])
        else:
            tab1, tab2, tab3 = st.tabs([
                "📊 Log Progress", 
                "📈 Progress History", 
                "📋 Summary"
            ])
        
        with tab1:
            _render_progress_entry_form(progress_tracker, user_id, user_goal)
        
        with tab2:
            _render_progress_history(progress_tracker, user_id, user_goal)
        
        with tab3:
            _render_progress_summary(progress_tracker, user_id, user_goal)
        
        if show_visualization:
            with tab4:
                _render_progress_visualization(progress_tracker, user_id, user_goal)


def _render_progress_entry_form(progress_tracker, user_id: str, user_goal) -> None:
    """
    Render the progress entry form with weight and measurement inputs.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        user_goal: User's current fitness goal
    """
    try:
        st.subheader("📊 Log Your Progress")
        
        # Progress entry form
        with st.form("progress_entry_form", clear_on_submit=True):
            st.write("Record your current measurements and achievements:")
            
            # Date selection
            entry_date = st.date_input(
                "Date:",
                value=datetime.now().date(),
                max_value=datetime.now().date(),
                help="Select the date for this progress entry"
            )
            
            # Weight input (primary measurement)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                weight = st.number_input(
                    "Weight (kg):",
                    min_value=30.0,
                    max_value=300.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    help="Enter your current weight in kilograms"
                )
            
            with col2:
                if user_goal:
                    target_weight = user_goal.target_value
                    st.metric(
                        "Target", 
                        f"{target_weight:.1f} kg",
                        delta=f"{weight - target_weight:.1f} kg" if weight else None
                    )
            
            # Body measurements section
            st.write("**Body Measurements (optional):**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                waist = st.number_input(
                    "Waist (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
                
                chest = st.number_input(
                    "Chest (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
            
            with col2:
                arms = st.number_input(
                    "Arms (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
                
                thighs = st.number_input(
                    "Thighs (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
            
            with col3:
                hips = st.number_input(
                    "Hips (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
                
                neck = st.number_input(
                    "Neck (cm):",
                    min_value=0.0,
                    max_value=200.0,
                    value=None,
                    step=0.5,
                    format="%.1f"
                )
            
            # Workout achievements section
            st.write("**Workout Achievements (optional):**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                workout_duration = st.number_input(
                    "Workout Duration (minutes):",
                    min_value=0,
                    max_value=480,
                    value=None,
                    step=5,
                    help="Total workout time in minutes"
                )
                
                exercises_completed = st.number_input(
                    "Exercises Completed:",
                    min_value=0,
                    max_value=50,
                    value=None,
                    step=1,
                    help="Number of different exercises performed"
                )
            
            with col2:
                calories_burned = st.number_input(
                    "Calories Burned:",
                    min_value=0,
                    max_value=2000,
                    value=None,
                    step=10,
                    help="Estimated calories burned during workout"
                )
            
            # Notes section
            notes = st.text_area(
                "Notes (optional):",
                placeholder="How are you feeling? Any observations about your progress?",
                max_chars=500,
                help="Add any notes about your progress, how you're feeling, or observations"
            )
            
            # Form submission
            submitted = st.form_submit_button("💾 Save Progress Entry", type="primary")
            
            if submitted:
                _handle_progress_submission(
                    progress_tracker, user_id, entry_date, weight,
                    {
                        'waist': waist, 'chest': chest, 'arms': arms,
                        'thighs': thighs, 'hips': hips, 'neck': neck
                    },
                    {
                        'duration_minutes': workout_duration,
                        'exercises_completed': exercises_completed,
                        'calories_burned': calories_burned
                    },
                    notes
                )
        
        # Quick entry buttons for common scenarios
        st.write("**Quick Entry:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⚡ Weight Only", help="Log just your weight"):
                st.session_state.quick_entry_mode = 'weight_only'
                st.rerun()
        
        with col2:
            if st.button("💪 Post-Workout", help="Log workout achievements"):
                st.session_state.quick_entry_mode = 'workout'
                st.rerun()
        
        with col3:
            if st.button("📏 Full Measurements", help="Log all measurements"):
                st.session_state.quick_entry_mode = 'full'
                st.rerun()
                
    except Exception as e:
        logger.error(f"Error rendering progress entry form: {str(e)}")
        st.error("Unable to display progress entry form.")


def _handle_progress_submission(
    progress_tracker, user_id: str, entry_date, weight: Optional[float],
    body_measurements: Dict[str, Optional[float]],
    workout_achievements: Dict[str, Optional[int]],
    notes: Optional[str]
) -> None:
    """
    Handle progress form submission with comprehensive validation and saving.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        entry_date: Date of the progress entry
        weight: Weight measurement
        body_measurements: Dictionary of body measurements
        workout_achievements: Dictionary of workout data
        notes: User notes
    """
    # Comprehensive input validation
    validation_errors = []
    
    # Validate date
    if entry_date > datetime.now().date():
        show_validation_error("Date", "Cannot log progress for future dates", 
                            ["Select today's date or an earlier date"])
        validation_errors.append("date")
    
    # Validate weight
    if weight is not None and not validate_weight_input(weight, "Weight"):
        validation_errors.append("weight")
    
    # Validate body measurements
    measurement_names = {
        'waist': 'Waist', 'chest': 'Chest', 'arms': 'Arms',
        'thighs': 'Thighs', 'hips': 'Hips', 'neck': 'Neck'
    }
    
    for key, value in body_measurements.items():
        if value is not None and not validate_measurement_input(value, measurement_names.get(key, key)):
            validation_errors.append(f"measurement_{key}")
    
    # Validate workout achievements
    if workout_achievements.get('duration_minutes') is not None:
        duration = workout_achievements['duration_minutes']
        if duration < 0 or duration > 480:  # 8 hours max
            show_validation_error("Workout Duration", "Must be between 0 and 480 minutes", 
                                ["Enter a realistic workout duration"])
            validation_errors.append("workout_duration")
    
    if workout_achievements.get('exercises_completed') is not None:
        exercises = workout_achievements['exercises_completed']
        if exercises < 0 or exercises > 50:
            show_validation_error("Exercises Completed", "Must be between 0 and 50", 
                                ["Enter a realistic number of exercises"])
            validation_errors.append("workout_exercises")
    
    if workout_achievements.get('calories_burned') is not None:
        calories = workout_achievements['calories_burned']
        if calories < 0 or calories > 2000:
            show_validation_error("Calories Burned", "Must be between 0 and 2000", 
                                ["Enter a realistic calorie count"])
            validation_errors.append("workout_calories")
    
    # Validate notes length
    if notes and len(notes.strip()) > 500:
        show_validation_error("Notes", "Must be 500 characters or less", 
                            [f"Current length: {len(notes)} characters", 
                             "Shorten your notes"])
        validation_errors.append("notes")
    
    # Stop if there are validation errors
    if validation_errors:
        show_error("Please fix the validation errors above before saving your progress")
        return
    
    # Prepare progress data
    progress_data = {
        'date': datetime.combine(entry_date, datetime.min.time())
    }
    
    # Add weight if provided
    if weight is not None:
        progress_data['weight'] = weight
    
    # Add body measurements if any provided
    filtered_measurements = {k: v for k, v in body_measurements.items() 
                           if v is not None and v > 0}
    if filtered_measurements:
        progress_data['body_measurements'] = filtered_measurements
    
    # Add workout achievements if any provided
    filtered_workouts = {k: v for k, v in workout_achievements.items() 
                        if v is not None and v > 0}
    if filtered_workouts:
        progress_data['workout_achievements'] = filtered_workouts
    
    # Add notes if provided
    if notes and notes.strip():
        progress_data['notes'] = notes.strip()
    
    # Validate that at least some data is provided
    if not any([
        weight is not None,
        filtered_measurements,
        filtered_workouts,
        notes and notes.strip()
    ]):
        show_validation_error("Progress Entry", 
                            "Please enter at least one measurement, workout achievement, or note",
                            ["Add your weight", "Record body measurements", 
                             "Log workout details", "Add notes about your progress"])
        return
    
    # Save the progress entry with comprehensive error handling
    try:
        with safe_operation("save progress entry", 
                          loading_message="Saving your progress...",
                          success_message="Progress entry saved successfully! 🎉"):
            
            entry = progress_tracker.log_progress(user_id, progress_data)
            
            # Show detailed success information
            success_details = []
            if weight is not None:
                success_details.append(f"Weight: {weight:.1f} kg")
            if filtered_measurements:
                success_details.append(f"{len(filtered_measurements)} body measurements")
            if filtered_workouts:
                success_details.append("workout achievements")
            if notes and notes.strip():
                success_details.append("notes")
            
            if success_details:
                show_info(f"Recorded: {', '.join(success_details)}")
            
            # Check for milestones or achievements
            _check_progress_milestones(progress_tracker, user_id, weight, entry_date)
            
            # Log success
            logger.info(f"Progress entry saved for user {user_id}: {entry.entry_id}")
            
            # Rerun to refresh displays
            st.rerun()
    
    except ProgressTrackerError as e:
        show_error("Failed to save progress entry", e, recoverable=True)
    except Exception as e:
        show_error("Unexpected error while saving progress", e, recoverable=True)


def _check_progress_milestones(progress_tracker, user_id: str, 
                              weight: Optional[float], entry_date) -> None:
    """
    Check for progress milestones and show congratulatory messages.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        weight: Current weight entry
        entry_date: Date of the entry
    """
    try:
        if weight is None:
            return
        
        # Get progress history to check for milestones
        history = progress_tracker.get_progress_history(user_id)
        weight_entries = [entry for entry in history if entry.weight is not None]
        
        if len(weight_entries) <= 1:
            show_success("Congratulations on logging your first weight entry! 🎉", 
                        "Keep tracking regularly to see your progress trends.")
            return
        
        # Check for consistency milestones
        if len(weight_entries) == 7:
            show_success("Amazing! You've logged 7 weight entries! 🌟", 
                        "Consistent tracking is key to reaching your goals.")
        elif len(weight_entries) == 30:
            show_success("Incredible! 30 progress entries logged! 🏆", 
                        "Your dedication to tracking is paying off!")
        
        # Check for weight change milestones
        if len(weight_entries) >= 2:
            first_weight = weight_entries[-1].weight  # Oldest entry (list is reversed)
            current_weight = weight
            weight_change = current_weight - first_weight
            
            # Milestone thresholds
            milestones = [1.0, 2.5, 5.0, 10.0, 15.0, 20.0]
            
            for milestone in milestones:
                if abs(weight_change) >= milestone:
                    direction = "lost" if weight_change < 0 else "gained"
                    show_success(f"Milestone achieved! You've {direction} {abs(weight_change):.1f} kg! 🎯", 
                               "Keep up the excellent work!")
                    break
    
    except Exception as e:
        logger.warning(f"Error checking progress milestones: {str(e)}")
        # Don't show error to user as this is not critical


def _render_progress_history(progress_tracker, user_id: str, user_goal) -> None:
    """
    Render progress history display with tabular format.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        user_goal: User's current fitness goal
    """
    try:
        st.subheader("📈 Progress History")
        
        # Load progress history
        try:
            history = progress_tracker.get_progress_history(user_id)
        except ProgressTrackerError as e:
            st.error(f"Error loading progress history: {str(e)}")
            return
        
        if not history:
            st.info("📝 No progress entries yet. Start logging your progress to see your history!")
            return
        
        # Filter and display options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Date range filter
            if len(history) > 1:
                date_range = st.selectbox(
                    "Show entries from:",
                    options=['All time', 'Last 30 days', 'Last 7 days', 'This week'],
                    index=0
                )
                
                # Filter history based on selection
                if date_range != 'All time':
                    cutoff_date = datetime.now()
                    if date_range == 'Last 30 days':
                        cutoff_date -= timedelta(days=30)
                    elif date_range == 'Last 7 days':
                        cutoff_date -= timedelta(days=7)
                    elif date_range == 'This week':
                        cutoff_date -= timedelta(days=cutoff_date.weekday())
                        cutoff_date = cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    history = [entry for entry in history if entry.date >= cutoff_date]
        
        with col2:
            # Entry type filter
            entry_types = st.multiselect(
                "Show entries with:",
                options=['Weight', 'Measurements', 'Workouts', 'Notes'],
                default=['Weight', 'Measurements', 'Workouts', 'Notes']
            )
        
        with col3:
            # Limit number of entries
            max_entries = st.selectbox(
                "Max entries:",
                options=[10, 25, 50, 100, 'All'],
                index=1
            )
        
        # Apply filters
        filtered_history = []
        for entry in history:
            include_entry = False
            
            if 'Weight' in entry_types and entry.weight is not None:
                include_entry = True
            if 'Measurements' in entry_types and entry.body_measurements:
                include_entry = True
            if 'Workouts' in entry_types and entry.workout_achievements:
                include_entry = True
            if 'Notes' in entry_types and entry.notes:
                include_entry = True
            
            if include_entry:
                filtered_history.append(entry)
        
        # Limit entries if specified
        if max_entries != 'All':
            filtered_history = filtered_history[:max_entries]
        
        if not filtered_history:
            st.info("No entries match the selected filters.")
            return
        
        # Display progress table
        _display_progress_table(filtered_history, user_goal)
        
        # Export options
        if len(filtered_history) > 0:
            st.write("**Export Options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download as CSV"):
                    csv_data = _export_progress_to_csv(filtered_history)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"fitness_progress_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear All Data"):
                    st.session_state.show_clear_confirmation = True
                    st.rerun()
        
        # Clear data confirmation
        if st.session_state.get('show_clear_confirmation', False):
            st.warning("⚠️ Are you sure you want to delete all progress data? This cannot be undone.")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Clear All", key="confirm_clear"):
                    try:
                        data_storage = get_data_storage()
                        # Clear only progress history, keep goal
                        if 'progress_history' in st.session_state:
                            if user_id in st.session_state.progress_history:
                                del st.session_state.progress_history[user_id]
                        st.success("All progress data cleared successfully!")
                        st.session_state.show_clear_confirmation = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {str(e)}")
            
            with col2:
                if st.button("Cancel", key="cancel_clear"):
                    st.session_state.show_clear_confirmation = False
                    st.rerun()
                    
    except Exception as e:
        logger.error(f"Error rendering progress history: {str(e)}")
        st.error("Unable to display progress history.")


def _display_progress_table(history: List[ProgressEntry], user_goal) -> None:
    """
    Display progress entries in a formatted table.
    
    Args:
        history: List of ProgressEntry instances
        user_goal: User's current fitness goal
    """
    try:
        # Prepare data for table display
        table_data = []
        
        for entry in history:
            row = {
                'Date': entry.date.strftime('%Y-%m-%d'),
                'Weight (kg)': f"{entry.weight:.1f}" if entry.weight else "-"
            }
            
            # Add body measurements
            if entry.body_measurements:
                measurements = entry.body_measurements
                row['Waist (cm)'] = f"{measurements.get('waist', 0):.1f}" if measurements.get('waist') else "-"
                row['Chest (cm)'] = f"{measurements.get('chest', 0):.1f}" if measurements.get('chest') else "-"
                row['Arms (cm)'] = f"{measurements.get('arms', 0):.1f}" if measurements.get('arms') else "-"
            else:
                row['Waist (cm)'] = "-"
                row['Chest (cm)'] = "-"
                row['Arms (cm)'] = "-"
            
            # Add workout info
            if entry.workout_achievements:
                workouts = entry.workout_achievements
                duration = workouts.get('duration_minutes', 0)
                calories = workouts.get('calories_burned', 0)
                row['Workout'] = f"{duration}min" if duration else "-"
                row['Calories'] = f"{calories}" if calories else "-"
            else:
                row['Workout'] = "-"
                row['Calories'] = "-"
            
            # Add notes (truncated)
            if entry.notes:
                notes_preview = entry.notes[:30] + "..." if len(entry.notes) > 30 else entry.notes
                row['Notes'] = notes_preview
            else:
                row['Notes'] = "-"
            
            # Calculate progress vs goal
            if entry.weight and user_goal:
                progress_pct = user_goal.get_progress_percentage(entry.weight)
                row['Progress %'] = f"{progress_pct:.1f}%"
            else:
                row['Progress %'] = "-"
            
            table_data.append(row)
        
        # Create and display DataFrame
        if table_data:
            df = pd.DataFrame(table_data)
            
            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Date': st.column_config.DateColumn('Date'),
                    'Weight (kg)': st.column_config.NumberColumn('Weight (kg)'),
                    'Progress %': st.column_config.ProgressColumn('Progress %', min_value=0, max_value=100),
                }
            )
            
            st.caption(f"Showing {len(table_data)} entries")
        
    except Exception as e:
        logger.error(f"Error displaying progress table: {str(e)}")
        st.error("Unable to display progress table.")


def _render_progress_summary(progress_tracker, user_id: str, user_goal) -> None:
    """
    Render basic progress summary statistics display.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        user_goal: User's current fitness goal
    """
    try:
        st.subheader("📋 Progress Summary")
        
        # Get comprehensive progress summary
        try:
            summary = progress_tracker.get_progress_summary(user_id)
        except ProgressTrackerError as e:
            st.error(f"Error loading progress summary: {str(e)}")
            return
        
        if not summary['has_goal']:
            st.info("Set a fitness goal to see progress summary.")
            return
        
        # Goal overview
        st.write("**🎯 Goal Overview**")
        goal_summary = summary['goal_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Goal",
                summary['goal_type'].replace('_', ' ').title()
            )
        
        with col2:
            st.metric(
                "Target Weight",
                f"{goal_summary['target_value']:.1f} kg"
            )
        
        with col3:
            days_remaining = goal_summary['days_remaining']
            if days_remaining > 0:
                st.metric("Days Remaining", days_remaining)
            else:
                if goal_summary['is_overdue']:
                    st.metric("Status", "Overdue", delta="⚠️")
                else:
                    st.metric("Status", "Complete", delta="🎉")
        
        with col4:
            progress_pct = summary['trend_analysis']['progress_percentage']
            st.metric("Progress", f"{progress_pct:.1f}%")
        
        st.divider()
        
        # Trend analysis
        st.write("**📈 Trend Analysis**")
        trend = summary['trend_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status indicator
            status = trend['status']
            status_colors = {
                'ahead': '🟢',
                'on_track': '🟡', 
                'behind': '🔴'
            }
            status_text = {
                'ahead': 'Ahead of Schedule',
                'on_track': 'On Track',
                'behind': 'Behind Schedule'
            }
            
            st.metric(
                "Status",
                f"{status_colors.get(status, '⚪')} {status_text.get(status, status)}"
            )
            
            # Current rate
            current_rate = trend['current_rate']
            required_rate = trend['required_rate']
            
            st.metric(
                "Current Rate",
                f"{abs(current_rate):.2f} kg/week",
                delta=f"{current_rate - required_rate:+.2f} vs required"
            )
        
        with col2:
            # Trend direction
            direction = trend['trend_direction']
            direction_icons = {
                'improving': '📈',
                'declining': '📉',
                'stable': '➡️'
            }
            
            st.metric(
                "Trend Direction",
                f"{direction_icons.get(direction, '➡️')} {direction.title()}"
            )
            
            # Confidence level
            confidence = trend['confidence']
            st.metric(
                "Analysis Confidence",
                f"{confidence * 100:.0f}%"
            )
        
        # Trend message
        st.info(trend['message'])
        
        st.divider()
        
        # Statistics
        st.write("**📊 Statistics**")
        stats = summary['statistics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", stats['total_entries'])
        
        with col2:
            st.metric("Weight Entries", stats['entries_with_weight'])
        
        with col3:
            st.metric("Measurement Entries", stats['entries_with_measurements'])
        
        with col4:
            st.metric("Workout Entries", stats['entries_with_workouts'])
        
        # Additional statistics
        if stats['average_weight']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Weight", f"{stats['average_weight']:.1f} kg")
            
            with col2:
                if stats['most_recent_weight']:
                    st.metric("Latest Weight", f"{stats['most_recent_weight']:.1f} kg")
            
            with col3:
                if stats['weight_change_total']:
                    change = stats['weight_change_total']
                    st.metric(
                        "Total Change",
                        f"{abs(change):.1f} kg",
                        delta=f"{change:+.1f} kg"
                    )
        
        # Recent entries preview
        if summary['recent_entries']:
            st.write("**📝 Recent Entries**")
            
            recent_df_data = []
            for entry in summary['recent_entries'][:5]:  # Show last 5
                recent_df_data.append({
                    'Date': entry['date'][:10],  # YYYY-MM-DD part
                    'Weight': f"{entry['weight']:.1f} kg" if entry['weight'] else "-",
                    'Notes': entry['notes'][:50] + "..." if entry['notes'] and len(entry['notes']) > 50 else entry['notes'] or "-"
                })
            
            if recent_df_data:
                recent_df = pd.DataFrame(recent_df_data)
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
        # Data quality indicators
        st.write("**🔍 Data Quality**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tracking_days = stats['date_range_days']
            if tracking_days > 0:
                entry_frequency = stats['total_entries'] / max(1, tracking_days / 7)  # entries per week
                st.metric("Tracking Period", f"{tracking_days} days")
                st.metric("Entry Frequency", f"{entry_frequency:.1f}/week")
        
        with col2:
            if summary['last_updated']:
                last_update = datetime.fromisoformat(summary['last_updated'])
                days_since = (datetime.now() - last_update).days
                st.metric("Last Entry", f"{days_since} days ago")
                
                if days_since > 7:
                    st.warning("⚠️ Consider logging progress more frequently for better tracking.")
                elif days_since <= 1:
                    st.success("✅ Great job staying consistent with tracking!")
                    
    except Exception as e:
        logger.error(f"Error rendering progress summary: {str(e)}")
        st.error("Unable to display progress summary.")


def _export_progress_to_csv(history: List[ProgressEntry]) -> str:
    """
    Export progress history to CSV format.
    
    Args:
        history: List of ProgressEntry instances
        
    Returns:
        CSV data as string
    """
    try:
        # Prepare data for CSV export
        csv_data = []
        
        for entry in history:
            row = {
                'Date': entry.date.strftime('%Y-%m-%d %H:%M:%S'),
                'Weight_kg': entry.weight if entry.weight else '',
                'Notes': entry.notes if entry.notes else ''
            }
            
            # Add body measurements
            if entry.body_measurements:
                for measurement, value in entry.body_measurements.items():
                    row[f'{measurement}_cm'] = value
            
            # Add workout achievements
            if entry.workout_achievements:
                for achievement, value in entry.workout_achievements.items():
                    row[f'workout_{achievement}'] = value
            
            csv_data.append(row)
        
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error exporting progress to CSV: {str(e)}")
        return ""


def _render_progress_visualization(progress_tracker, user_id: str, user_goal) -> None:
    """
    Render progress visualization with charts and trend analysis.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_id: User identifier
        user_goal: User's current fitness goal
    """
    try:
        st.subheader("📊 Progress Visualization")
        
        # Load progress history
        try:
            history = progress_tracker.get_progress_history(user_id)
        except ProgressTrackerError as e:
            st.error(f"Error loading progress data: {str(e)}")
            return
        
        # Filter weight entries
        weight_entries = [entry for entry in history if entry.weight is not None]
        
        if len(weight_entries) < 2:
            st.info("📈 Need at least 2 weight entries to generate meaningful visualizations. Keep logging your progress!")
            return
        
        # Chart generation options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Chart Options:**")
            
            # Time range selector
            time_ranges = {
                'All Time': None,
                'Last 30 Days': 30,
                'Last 60 Days': 60,
                'Last 90 Days': 90
            }
            
            selected_range = st.selectbox(
                "Time Range:",
                options=list(time_ranges.keys()),
                index=0
            )
            
            # Filter data based on time range
            filtered_entries = weight_entries
            if time_ranges[selected_range] is not None:
                cutoff_date = datetime.now() - timedelta(days=time_ranges[selected_range])
                filtered_entries = [entry for entry in weight_entries if entry.date >= cutoff_date]
            
            # Show data points info
            st.metric("Data Points", len(filtered_entries))
            
            if filtered_entries:
                date_range = (filtered_entries[-1].date - filtered_entries[0].date).days
                st.metric("Date Range", f"{date_range} days")
        
        with col1:
            # Generate and display chart
            try:
                chart_fig = progress_tracker.generate_progress_chart(user_goal, filtered_entries)
                
                if chart_fig is not None:
                    st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    _render_fallback_visualization(filtered_entries, user_goal)
                    
            except Exception as e:
                logger.error(f"Error generating progress chart: {str(e)}")
                st.error("Unable to generate chart. Showing alternative visualization.")
                _render_fallback_visualization(filtered_entries, user_goal)
        
        # Trend analysis section
        if len(filtered_entries) >= 2:
            st.divider()
            _render_trend_analysis_section(progress_tracker, user_goal, filtered_entries)
        
        # Progress insights
        st.divider()
        _render_progress_insights(filtered_entries, user_goal)
        
    except Exception as e:
        logger.error(f"Error rendering progress visualization: {str(e)}")
        st.error("Unable to display progress visualization.")


def _render_fallback_visualization(weight_entries: List[ProgressEntry], user_goal) -> None:
    """
    Render fallback visualization when Plotly is not available.
    
    Args:
        weight_entries: List of progress entries with weight data
        user_goal: User's current fitness goal
    """
    try:
        st.warning("📊 Advanced charts not available. Showing basic visualization.")
        
        # Create simple line chart using Streamlit's built-in charting
        chart_data = pd.DataFrame({
            'Date': [entry.date.date() for entry in weight_entries],
            'Weight': [entry.weight for entry in weight_entries],
            'Target': [user_goal.target_value] * len(weight_entries)
        })
        
        chart_data = chart_data.set_index('Date')
        
        st.line_chart(chart_data)
        
        # Show basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_weight = weight_entries[-1].weight
            st.metric("Current Weight", f"{current_weight:.1f} kg")
        
        with col2:
            weight_change = weight_entries[-1].weight - weight_entries[0].weight
            st.metric("Total Change", f"{weight_change:+.1f} kg")
        
        with col3:
            progress_pct = user_goal.get_progress_percentage(current_weight)
            st.metric("Progress", f"{progress_pct:.1f}%")
        
    except Exception as e:
        logger.error(f"Error rendering fallback visualization: {str(e)}")
        st.error("Unable to display visualization.")


def _render_trend_analysis_section(progress_tracker, user_goal, weight_entries: List[ProgressEntry]) -> None:
    """
    Render detailed trend analysis section.
    
    Args:
        progress_tracker: ProgressTracker instance
        user_goal: User's current fitness goal
        weight_entries: List of progress entries with weight data
    """
    try:
        st.write("**📈 Trend Analysis**")
        
        # Calculate trend analysis
        trend = progress_tracker.calculate_trend(user_goal, weight_entries)
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_icons = {
                'ahead': '🟢',
                'on_track': '🟡',
                'behind': '🔴'
            }
            status_text = {
                'ahead': 'Ahead',
                'on_track': 'On Track',
                'behind': 'Behind'
            }
            
            st.metric(
                "Status",
                f"{status_icons.get(trend.status, '⚪')} {status_text.get(trend.status, 'Unknown')}"
            )
        
        with col2:
            direction_icons = {
                'improving': '📈',
                'declining': '📉',
                'stable': '➡️'
            }
            
            st.metric(
                "Trend",
                f"{direction_icons.get(trend.trend_direction, '➡️')} {trend.trend_direction.title()}"
            )
        
        with col3:
            st.metric(
                "Current Rate",
                f"{abs(trend.current_rate):.2f} kg/week"
            )
        
        with col4:
            st.metric(
                "Confidence",
                f"{trend.confidence * 100:.0f}%"
            )
        
        # Detailed analysis
        st.info(trend.message)
        
        # Rate comparison
        if trend.required_rate != 0:
            rate_diff = abs(trend.current_rate) - abs(trend.required_rate)
            
            if abs(rate_diff) > 0.1:  # Significant difference
                if rate_diff > 0:
                    st.success(f"✅ You're progressing {rate_diff:.2f} kg/week faster than required!")
                else:
                    st.warning(f"⚠️ You need to increase your rate by {abs(rate_diff):.2f} kg/week to meet your goal.")
        
    except Exception as e:
        logger.error(f"Error rendering trend analysis: {str(e)}")
        st.error("Unable to display trend analysis.")


def _render_progress_insights(weight_entries: List[ProgressEntry], user_goal) -> None:
    """
    Render progress insights and recommendations.
    
    Args:
        weight_entries: List of progress entries with weight data
        user_goal: User's current fitness goal
    """
    try:
        st.write("**💡 Progress Insights**")
        
        # Calculate insights
        weights = [entry.weight for entry in weight_entries]
        dates = [entry.date for entry in weight_entries]
        
        # Weight change analysis
        total_change = weights[-1] - weights[0]
        days_tracked = (dates[-1] - dates[0]).days
        
        insights = []
        
        # Progress consistency
        if len(weights) >= 4:
            # Calculate variance to assess consistency
            weight_variance = sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights)
            
            if weight_variance < 1.0:
                insights.append("🎯 Your progress is very consistent - great job maintaining steady habits!")
            elif weight_variance < 4.0:
                insights.append("📊 Your progress shows good consistency with normal fluctuations.")
            else:
                insights.append("📈 Your weight shows significant fluctuations. Consider tracking daily habits for better consistency.")
        
        # Rate of progress
        if days_tracked > 0:
            weekly_rate = abs(total_change) / (days_tracked / 7)
            
            if user_goal.goal_type == 'weight_loss':
                if weekly_rate > 1.0:
                    insights.append("⚡ You're losing weight quickly! Make sure you're eating enough to maintain energy.")
                elif weekly_rate > 0.5:
                    insights.append("✅ Excellent weight loss rate - sustainable and healthy!")
                elif weekly_rate > 0.2:
                    insights.append("🐌 Slow and steady progress - this is sustainable long-term!")
                else:
                    insights.append("🔍 Progress is very slow. Consider reviewing your nutrition and exercise plan.")
            
            elif user_goal.goal_type == 'muscle_building':
                if weekly_rate > 0.5:
                    insights.append("💪 Great muscle building progress! Make sure you're getting enough protein.")
                elif weekly_rate > 0.2:
                    insights.append("✅ Solid muscle building rate - keep up the strength training!")
                else:
                    insights.append("🏋️ Consider increasing protein intake and progressive overload in workouts.")
        
        # Milestone achievements
        progress_pct = user_goal.get_progress_percentage(weights[-1])
        
        if progress_pct >= 75:
            insights.append("🏆 You're in the final stretch! Stay focused to reach your goal!")
        elif progress_pct >= 50:
            insights.append("🎉 Halfway there! Your consistency is paying off!")
        elif progress_pct >= 25:
            insights.append("🚀 Great start! You've made solid progress toward your goal!")
        
        # Time remaining insights
        days_remaining = user_goal.days_remaining()
        
        if days_remaining <= 7:
            insights.append("⏰ Final week! Focus on maintaining your current habits.")
        elif days_remaining <= 30:
            insights.append("📅 One month left - time to push through any plateaus!")
        
        # Display insights
        for insight in insights:
            st.write(f"• {insight}")
        
        if not insights:
            st.write("• Keep logging your progress to unlock personalized insights!")
        
    except Exception as e:
        logger.error(f"Error rendering progress insights: {str(e)}")
        st.error("Unable to display progress insights.")


def initialize_progress_ui_session_state() -> None:
    """
    Initialize session state variables for the progress UI.
    """
    if 'quick_entry_mode' not in st.session_state:
        st.session_state.quick_entry_mode = None
    
    if 'show_clear_confirmation' not in st.session_state:
        st.session_state.show_clear_confirmation = False