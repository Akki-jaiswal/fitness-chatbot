"""
Goal setting user interface components for Streamlit.

This module provides the UI components for setting and managing fitness goals
in the Streamlit sidebar with goal type selection, input fields, and summary display.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from .goal_manager import get_goal_manager, GoalManagerError
from .models import FitnessGoal
from .data_storage import get_data_storage
from .data_management_ui import render_data_management_sidebar
from .ui_feedback import (
    show_success, show_error, show_warning, show_info, show_validation_error,
    loading_indicator, confirm_action, safe_operation, handle_ui_errors,
    validate_weight_input, validate_timeframe_input
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalUIError(Exception):
    """Custom exception for goal UI operations."""
    pass


@handle_ui_errors("render goal setting interface")
def render_goal_setting_sidebar() -> None:
    """
    Render the complete goal setting interface in the Streamlit sidebar.
    
    This function creates all the UI components for goal management including:
    - Goal type selection with radio buttons
    - Input fields for current/target values and timeframe
    - Goal summary display and update functionality
    """
    with safe_operation("load goal interface", 
                       loading_message="Loading goal interface..."):
        # Initialize data storage and goal manager
        data_storage = get_data_storage()
        goal_manager = get_goal_manager()
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Create sidebar section for goal management
        with st.sidebar:
            st.header("🎯 Fitness Goals")
            
            # Load existing goal with error handling
            existing_goal = None
            try:
                with loading_indicator("Loading your goal...", "load_goal"):
                    existing_goal = goal_manager.get_user_goal(user_id)
            except GoalManagerError as e:
                show_error("Failed to load your goal", e, recoverable=True)
            except Exception as e:
                show_error("Unexpected error loading goal", e, recoverable=True)
            
            # Display goal summary if exists
            if existing_goal and existing_goal.is_active:
                _render_goal_summary(existing_goal, goal_manager, user_id)
                st.divider()
            
            # Goal setting form
            _render_goal_setting_form(existing_goal, goal_manager, user_id)
            
            # Data management sidebar widget
            st.divider()
            render_data_management_sidebar()


@handle_ui_errors("display goal summary")
def _render_goal_summary(goal: FitnessGoal, goal_manager, user_id: str) -> None:
    """
    Render the goal summary display section.
    
    Args:
        goal: Current FitnessGoal instance
        goal_manager: GoalManager instance
        user_id: User identifier
    """
    st.subheader("📊 Current Goal")
    
    # Get goal summary with progress information
    try:
        with loading_indicator("Loading goal summary...", "goal_summary"):
            summary = goal_manager.get_goal_summary(user_id)
    except Exception as e:
        show_error("Failed to load goal summary", e, recoverable=True)
        return
    
    if summary:
        # Goal type with emoji
        goal_type_display = {
            'weight_loss': '📉 Weight Loss',
            'muscle_building': '💪 Muscle Building', 
            'weight_maintenance': '⚖️ Weight Maintenance'
        }
        
        st.write(f"**Type:** {goal_type_display.get(goal.goal_type, goal.goal_type)}")
        
        # Current and target values
        if goal.goal_type == 'weight_maintenance':
            st.write(f"**Target Weight:** {goal.target_value:.1f} kg")
            st.write(f"**Current Weight:** {goal.current_value:.1f} kg")
        else:
            st.write(f"**Current:** {goal.current_value:.1f} kg")
            st.write(f"**Target:** {goal.target_value:.1f} kg")
            change_amount = abs(goal.target_value - goal.current_value)
            st.write(f"**Change Needed:** {change_amount:.1f} kg")
        
        # Timeline information with status indicators
        days_remaining = summary['days_remaining']
        if days_remaining > 0:
            st.write(f"**Time Remaining:** {days_remaining} days")
            
            # Progress bar for time with color coding
            time_progress = summary['time_progress_percentage']
            progress_color = "normal"
            if time_progress > 75:
                progress_color = "🟡"  # Warning - time running out
            elif time_progress > 90:
                progress_color = "🔴"  # Critical - almost out of time
            
            st.progress(time_progress / 100, text=f"Time Progress: {time_progress:.0f}% {progress_color}")
            
            # Show motivational messages based on progress
            if time_progress > 90:
                show_warning("Time is running out! Consider adjusting your goal or increasing efforts.")
            elif time_progress > 75:
                show_info("You're in the final stretch! Keep up the good work.")
        else:
            if summary['is_overdue']:
                show_warning("Your goal deadline has passed. Consider setting a new goal or extending the current one.")
            else:
                show_success("Congratulations! You've reached your goal deadline.")
        
        # Goal management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✏️ Update Goal", key="update_goal_btn", 
                        help="Modify your current goal"):
                st.session_state.show_goal_form = True
                st.session_state.editing_goal = True
                show_info("Goal editing mode activated")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Goal", key="delete_goal_btn", 
                        help="Permanently delete your current goal"):
                st.session_state.show_delete_confirmation = True
                st.rerun()
        
        # Enhanced delete confirmation dialog
        if st.session_state.get('show_delete_confirmation', False):
            if confirm_action(
                "Are you sure you want to delete your goal? This will also remove all associated progress data.",
                "delete_goal",
                danger=True
            ):
                try:
                    with safe_operation("delete goal", 
                                      loading_message="Deleting goal...",
                                      success_message="Goal deleted successfully!"):
                        goal_manager.delete_goal(user_id)
                        st.session_state.show_delete_confirmation = False
                        st.rerun()
                except GoalManagerError as e:
                    show_error("Failed to delete goal", e, recoverable=True)
                except Exception as e:
                    show_error("Unexpected error while deleting goal", e, recoverable=True)
            else:
                st.session_state.show_delete_confirmation = False
                st.rerun()


def _render_goal_setting_form(existing_goal: Optional[FitnessGoal], goal_manager, user_id: str) -> None:
    """
    Render the goal setting form with input fields.
    
    Args:
        existing_goal: Current FitnessGoal instance if exists
        goal_manager: GoalManager instance
        user_id: User identifier
    """
    try:
        # Determine if we should show the form
        show_form = (
            existing_goal is None or 
            st.session_state.get('show_goal_form', False) or
            st.session_state.get('editing_goal', False)
        )
        
        if not show_form:
            if st.button("➕ Set New Goal", key="new_goal_btn"):
                st.session_state.show_goal_form = True
                st.session_state.editing_goal = False
                st.rerun()
            return
        
        # Form header
        if existing_goal and st.session_state.get('editing_goal', False):
            st.subheader("✏️ Update Goal")
        else:
            st.subheader("➕ Set New Goal")
        
        # Goal type selection with radio buttons
        goal_type_options = {
            'weight_loss': '📉 Weight Loss',
            'muscle_building': '💪 Muscle Building',
            'weight_maintenance': '⚖️ Weight Maintenance'
        }
        
        # Default values from existing goal or defaults
        default_goal_type = existing_goal.goal_type if existing_goal else 'weight_loss'
        default_current = existing_goal.current_value if existing_goal else 70.0
        default_target = existing_goal.target_value if existing_goal else 65.0
        default_timeframe = existing_goal.timeframe_weeks if existing_goal else 12
        
        # Goal type selection
        selected_goal_type = st.radio(
            "Select your fitness goal:",
            options=list(goal_type_options.keys()),
            format_func=lambda x: goal_type_options[x],
            index=list(goal_type_options.keys()).index(default_goal_type),
            key="goal_type_radio"
        )
        
        # Dynamic help text based on goal type
        help_text = {
            'weight_loss': "Focus on creating a calorie deficit through diet and exercise.",
            'muscle_building': "Emphasize strength training and adequate protein intake.",
            'weight_maintenance': "Maintain current weight through balanced nutrition and exercise."
        }
        
        st.info(help_text[selected_goal_type])
        
        # Input fields for current and target values
        col1, col2 = st.columns(2)
        
        with col1:
            current_value = st.number_input(
                "Current Weight (kg):",
                min_value=30.0,
                max_value=300.0,
                value=float(default_current),
                step=0.1,
                format="%.1f",
                key="current_weight_input",
                help="Enter your current weight in kilograms"
            )
        
        with col2:
            # Adjust target value constraints based on goal type
            if selected_goal_type == 'weight_loss':
                max_target = current_value - 0.5  # Must be less than current
                target_help = "Target weight should be less than current weight"
            elif selected_goal_type == 'muscle_building':
                min_target = current_value + 0.5  # Must be more than current
                max_target = current_value + 20.0  # Reasonable upper limit
                target_help = "Target weight should be more than current weight"
            else:  # weight_maintenance
                min_target = current_value - 3.0  # Within 3kg range
                max_target = current_value + 3.0
                target_help = "Target should be within 3kg of current weight"
            
            # Set appropriate min/max values
            if selected_goal_type == 'weight_loss':
                target_value = st.number_input(
                    "Target Weight (kg):",
                    min_value=30.0,
                    max_value=max_target,
                    value=min(float(default_target), max_target),
                    step=0.1,
                    format="%.1f",
                    key="target_weight_input",
                    help=target_help
                )
            elif selected_goal_type == 'muscle_building':
                target_value = st.number_input(
                    "Target Weight (kg):",
                    min_value=min_target,
                    max_value=max_target,
                    value=max(float(default_target), min_target),
                    step=0.1,
                    format="%.1f",
                    key="target_weight_input",
                    help=target_help
                )
            else:  # weight_maintenance
                target_value = st.number_input(
                    "Target Weight (kg):",
                    min_value=min_target,
                    max_value=max_target,
                    value=max(min_target, min(float(default_target), max_target)),
                    step=0.1,
                    format="%.1f",
                    key="target_weight_input",
                    help=target_help
                )
        
        # Timeframe selection with slider
        timeframe_weeks = st.slider(
            "Timeframe (weeks):",
            min_value=4,
            max_value=52,
            value=default_timeframe,
            step=1,
            key="timeframe_slider",
            help="Choose a realistic timeframe for your goal (4-52 weeks)"
        )
        
        # Display calculated target date
        target_date = datetime.now() + timedelta(weeks=timeframe_weeks)
        st.write(f"**Target Date:** {target_date.strftime('%B %d, %Y')}")
        
        # Validation and goal preview
        weight_change = abs(target_value - current_value)
        weekly_change = weight_change / timeframe_weeks if timeframe_weeks > 0 else 0
        
        # Show goal preview
        st.write("**Goal Preview:**")
        if selected_goal_type == 'weight_maintenance':
            st.write(f"• Maintain weight around {target_value:.1f} kg")
        else:
            direction = "lose" if selected_goal_type == 'weight_loss' else "gain"
            st.write(f"• {direction.title()} {weight_change:.1f} kg in {timeframe_weeks} weeks")
            st.write(f"• Average: {weekly_change:.2f} kg per week")
        
        # Validation warnings
        if weekly_change > 1.0 and selected_goal_type != 'weight_maintenance':
            st.warning("⚠️ This rate of change may be too aggressive. Consider a longer timeframe.")
        elif weight_change < 2.0 and selected_goal_type != 'weight_maintenance':
            st.warning("⚠️ This change may be too small to be meaningful. Consider a larger target difference.")
        
        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Goal", key="save_goal_btn", type="primary"):
                _handle_goal_submission(
                    goal_manager, user_id, selected_goal_type, 
                    current_value, target_value, timeframe_weeks, existing_goal
                )
        
        with col2:
            if st.button("❌ Cancel", key="cancel_goal_btn"):
                st.session_state.show_goal_form = False
                st.session_state.editing_goal = False
                st.rerun()
        
    except Exception as e:
        logger.error(f"Error rendering goal setting form: {str(e)}")
        st.error("Unable to display goal setting form.")


def _handle_goal_submission(
    goal_manager, user_id: str, goal_type: str, 
    current_value: float, target_value: float, timeframe_weeks: int,
    existing_goal: Optional[FitnessGoal]
) -> None:
    """
    Handle goal form submission with comprehensive validation and saving.
    
    Args:
        goal_manager: GoalManager instance
        user_id: User identifier
        goal_type: Selected goal type
        current_value: Current weight value
        target_value: Target weight value
        timeframe_weeks: Goal timeframe in weeks
        existing_goal: Existing goal if updating
    """
    # Comprehensive input validation with user feedback
    validation_errors = []
    
    # Validate current weight
    if not validate_weight_input(current_value, "Current Weight"):
        validation_errors.append("current_weight")
    
    # Validate target weight
    if not validate_weight_input(target_value, "Target Weight"):
        validation_errors.append("target_weight")
    
    # Validate timeframe
    if not validate_timeframe_input(timeframe_weeks, "Timeframe"):
        validation_errors.append("timeframe")
    
    # Validate goal logic
    if current_value and target_value:
        if goal_type == 'weight_loss' and target_value >= current_value:
            show_validation_error("Target Weight", 
                                "Must be less than current weight for weight loss goals",
                                ["Reduce the target weight", 
                                 "Consider changing to muscle building or maintenance"])
            validation_errors.append("target_logic")
        
        elif goal_type == 'muscle_building' and target_value <= current_value:
            show_validation_error("Target Weight", 
                                "Must be greater than current weight for muscle building goals",
                                ["Increase the target weight", 
                                 "Consider changing to weight loss or maintenance"])
            validation_errors.append("target_logic")
        
        elif goal_type == 'weight_maintenance':
            weight_diff = abs(target_value - current_value)
            if weight_diff > 3.0:
                show_validation_error("Target Weight", 
                                    "Should be within 3kg of current weight for maintenance goals",
                                    ["Adjust target closer to current weight", 
                                     "Consider weight loss or muscle building instead"])
                validation_errors.append("maintenance_range")
    
    # Stop if there are validation errors
    if validation_errors:
        show_error("Please fix the validation errors above before saving your goal")
        return
    
    # Prepare goal data
    goal_data = {
        'goal_type': goal_type,
        'current_value': current_value,
        'target_value': target_value,
        'timeframe_weeks': timeframe_weeks
    }
    
    # Validate goal data with goal manager
    try:
        is_valid, error_msg = goal_manager.validate_goal_data(goal_data)
        if not is_valid:
            show_validation_error("Goal Data", error_msg, 
                                ["Check all input values", "Ensure realistic targets"])
            return
    except Exception as e:
        show_error("Failed to validate goal data", e, recoverable=True)
        return
    
    # Save the goal with comprehensive error handling
    operation_name = "update goal" if (existing_goal and st.session_state.get('editing_goal', False)) else "create goal"
    
    try:
        with safe_operation(operation_name, 
                          loading_message=f"Saving your {goal_type.replace('_', ' ')} goal...",
                          success_message=f"Goal {operation_name.split()[0]}d successfully! 🎉"):
            
            if existing_goal and st.session_state.get('editing_goal', False):
                # Update existing goal
                updates = {
                    'target_value': target_value,
                    'timeframe_weeks': timeframe_weeks
                }
                goal = goal_manager.update_goal(user_id, updates)
                
                # Show additional success details
                show_info(f"Updated target: {target_value:.1f} kg, timeframe: {timeframe_weeks} weeks")
            else:
                # Create new goal (replace existing if any)
                goal = goal_manager.replace_goal(user_id, goal_data)
                
                # Show goal details
                weight_change = abs(target_value - current_value)
                weekly_rate = weight_change / timeframe_weeks
                direction = "lose" if goal_type == 'weight_loss' else "gain" if goal_type == 'muscle_building' else "maintain"
                
                if goal_type != 'weight_maintenance':
                    show_info(f"Goal: {direction} {weight_change:.1f} kg in {timeframe_weeks} weeks (≈{weekly_rate:.2f} kg/week)")
                else:
                    show_info(f"Goal: maintain weight around {target_value:.1f} kg")
            
            # Reset form state
            st.session_state.show_goal_form = False
            st.session_state.editing_goal = False
            
            # Log success
            logger.info(f"Goal {operation_name} for user {user_id}: {goal_type}")
            
            # Rerun to refresh the display
            st.rerun()
    
    except GoalManagerError as e:
        show_error(f"Failed to {operation_name}", e, recoverable=True)
    except Exception as e:
        show_error(f"Unexpected error while {operation_name.replace('_', ' ')}", e, recoverable=True)


def get_user_goal_status() -> Dict[str, Any]:
    """
    Get the current user's goal status for use in other parts of the application.
    
    Returns:
        Dictionary with goal status information
    """
    try:
        goal_manager = get_goal_manager()
        user_id = st.session_state.get('user_id', 'default_user')
        
        goal = goal_manager.get_user_goal(user_id)
        
        if goal and goal.is_active:
            summary = goal_manager.get_goal_summary(user_id)
            return {
                'has_goal': True,
                'goal': goal,
                'summary': summary,
                'goal_type': goal.goal_type,
                'days_remaining': summary['days_remaining'] if summary else 0,
                'is_overdue': summary['is_overdue'] if summary else False
            }
        else:
            return {
                'has_goal': False,
                'goal': None,
                'summary': None,
                'goal_type': None,
                'days_remaining': 0,
                'is_overdue': False
            }
            
    except Exception as e:
        logger.error(f"Error getting user goal status: {str(e)}")
        return {
            'has_goal': False,
            'goal': None,
            'summary': None,
            'goal_type': None,
            'days_remaining': 0,
            'is_overdue': False,
            'error': str(e)
        }


def initialize_goal_ui_session_state() -> None:
    """
    Initialize session state variables for the goal UI.
    """
    if 'show_goal_form' not in st.session_state:
        st.session_state.show_goal_form = False
    
    if 'editing_goal' not in st.session_state:
        st.session_state.editing_goal = False
    
    if 'show_delete_confirmation' not in st.session_state:
        st.session_state.show_delete_confirmation = False