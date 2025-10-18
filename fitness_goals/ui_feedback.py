"""
User interface feedback and error handling utilities.

This module provides comprehensive error handling, loading indicators,
success notifications, and validation feedback for the fitness goals application.
"""

import streamlit as st
import time
import logging
from typing import Optional, Dict, Any, Callable, Union
from contextlib import contextmanager
from datetime import datetime
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class UIFeedbackManager:
    """
    Manages user interface feedback including errors, success messages,
    loading indicators, and validation feedback.
    """
    
    def __init__(self):
        """Initialize the feedback manager."""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for feedback management."""
        if 'ui_feedback_messages' not in st.session_state:
            st.session_state.ui_feedback_messages = []
        
        if 'ui_feedback_loading' not in st.session_state:
            st.session_state.ui_feedback_loading = {}
    
    @contextmanager
    def loading_indicator(self, message: str, key: Optional[str] = None):
        """
        Context manager for showing loading indicators during operations.
        
        Args:
            message: Loading message to display
            key: Optional unique key for the loading indicator
        """
        loading_key = key or f"loading_{int(time.time() * 1000)}"
        
        try:
            # Show loading indicator
            st.session_state.ui_feedback_loading[loading_key] = {
                'message': message,
                'start_time': datetime.now()
            }
            
            # Create placeholder for loading message
            placeholder = st.empty()
            with placeholder:
                with st.spinner(message):
                    yield placeholder
            
        finally:
            # Clear loading indicator
            if loading_key in st.session_state.ui_feedback_loading:
                del st.session_state.ui_feedback_loading[loading_key]
            
            # Clear placeholder
            if 'placeholder' in locals():
                placeholder.empty()
    
    def show_success(self, message: str, details: Optional[str] = None, auto_dismiss: bool = True):
        """
        Show a success message to the user.
        
        Args:
            message: Success message
            details: Optional additional details
            auto_dismiss: Whether to auto-dismiss the message
        """
        try:
            full_message = f"✅ {message}"
            if details:
                full_message += f"\n\n{details}"
            
            st.success(full_message)
            
            # Log success
            logger.info(f"Success: {message}")
            
            # Store in session state for persistence
            self._add_feedback_message('success', message, details)
            
            if auto_dismiss:
                # Auto-dismiss after a delay (simulated with session state)
                st.session_state.ui_feedback_auto_dismiss = {
                    'type': 'success',
                    'message': message,
                    'timestamp': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Error showing success message: {str(e)}")
    
    def show_error(self, message: str, error: Optional[Exception] = None, 
                   show_details: bool = False, recoverable: bool = True):
        """
        Show an error message to the user with appropriate formatting.
        
        Args:
            message: User-friendly error message
            error: Optional exception object
            show_details: Whether to show technical details
            recoverable: Whether the error is recoverable
        """
        try:
            # Format error message
            error_icon = "⚠️" if recoverable else "❌"
            full_message = f"{error_icon} {message}"
            
            # Add technical details if requested and available
            if show_details and error:
                full_message += f"\n\n**Technical details:** {str(error)}"
            
            # Show appropriate Streamlit message
            if recoverable:
                st.warning(full_message)
            else:
                st.error(full_message)
            
            # Log error with full details
            if error:
                logger.error(f"Error: {message} - {str(error)}")
                if show_details:
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.error(f"Error: {message}")
            
            # Store in session state
            self._add_feedback_message('error', message, str(error) if error else None)
            
            # Provide recovery suggestions for common errors
            if recoverable:
                self._show_recovery_suggestions(message, error)
        
        except Exception as e:
            # Fallback error handling
            st.error(f"❌ An unexpected error occurred: {message}")
            logger.error(f"Error in error handler: {str(e)}")
    
    def show_warning(self, message: str, details: Optional[str] = None):
        """
        Show a warning message to the user.
        
        Args:
            message: Warning message
            details: Optional additional details
        """
        try:
            full_message = f"⚠️ {message}"
            if details:
                full_message += f"\n\n{details}"
            
            st.warning(full_message)
            
            # Log warning
            logger.warning(f"Warning: {message}")
            
            # Store in session state
            self._add_feedback_message('warning', message, details)
        
        except Exception as e:
            logger.error(f"Error showing warning message: {str(e)}")
    
    def show_info(self, message: str, details: Optional[str] = None):
        """
        Show an informational message to the user.
        
        Args:
            message: Info message
            details: Optional additional details
        """
        try:
            full_message = f"ℹ️ {message}"
            if details:
                full_message += f"\n\n{details}"
            
            st.info(full_message)
            
            # Log info
            logger.info(f"Info: {message}")
            
            # Store in session state
            self._add_feedback_message('info', message, details)
        
        except Exception as e:
            logger.error(f"Error showing info message: {str(e)}")
    
    def show_validation_error(self, field_name: str, error_message: str, 
                            suggestions: Optional[list] = None):
        """
        Show validation error for form fields with suggestions.
        
        Args:
            field_name: Name of the field with validation error
            error_message: Validation error message
            suggestions: Optional list of suggestions to fix the error
        """
        try:
            message = f"**{field_name}:** {error_message}"
            
            if suggestions:
                message += "\n\n**Suggestions:**"
                for suggestion in suggestions:
                    message += f"\n• {suggestion}"
            
            st.error(f"❌ {message}")
            
            # Log validation error
            logger.warning(f"Validation error - {field_name}: {error_message}")
            
            # Store in session state
            self._add_feedback_message('validation_error', f"{field_name}: {error_message}", 
                                     suggestions)
        
        except Exception as e:
            logger.error(f"Error showing validation error: {str(e)}")
    
    def confirm_action(self, message: str, action_name: str, 
                      danger: bool = False) -> bool:
        """
        Show a confirmation dialog for user actions.
        
        Args:
            message: Confirmation message
            action_name: Name of the action to confirm
            danger: Whether this is a dangerous action
            
        Returns:
            True if user confirmed, False otherwise
        """
        try:
            # Create unique key for this confirmation
            confirm_key = f"confirm_{action_name}_{int(time.time())}"
            
            # Show confirmation message
            if danger:
                st.error(f"⚠️ **{message}**")
                st.write("This action cannot be undone.")
            else:
                st.warning(f"❓ **{message}**")
            
            # Create confirmation buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Yes, Continue", key=f"{confirm_key}_yes", 
                           type="primary" if not danger else "secondary"):
                    return True
            
            with col2:
                if st.button("❌ Cancel", key=f"{confirm_key}_no"):
                    return False
            
            return False
        
        except Exception as e:
            logger.error(f"Error showing confirmation dialog: {str(e)}")
            return False
    
    def _add_feedback_message(self, msg_type: str, message: str, details: Optional[str]):
        """Add feedback message to session state for persistence."""
        try:
            feedback_msg = {
                'type': msg_type,
                'message': message,
                'details': details,
                'timestamp': datetime.now().isoformat()
            }
            
            # Limit message history to prevent memory issues
            if len(st.session_state.ui_feedback_messages) >= 50:
                st.session_state.ui_feedback_messages = st.session_state.ui_feedback_messages[-25:]
            
            st.session_state.ui_feedback_messages.append(feedback_msg)
        
        except Exception as e:
            logger.error(f"Error storing feedback message: {str(e)}")
    
    def _show_recovery_suggestions(self, message: str, error: Optional[Exception]):
        """Show recovery suggestions based on error type."""
        try:
            suggestions = []
            
            # Common error patterns and suggestions
            if "validation" in message.lower():
                suggestions = [
                    "Check that all required fields are filled",
                    "Ensure numeric values are within valid ranges",
                    "Verify date formats are correct"
                ]
            elif "storage" in message.lower() or "save" in message.lower():
                suggestions = [
                    "Try refreshing the page and attempting again",
                    "Check your internet connection",
                    "Clear browser cache if the problem persists"
                ]
            elif "load" in message.lower() or "data" in message.lower():
                suggestions = [
                    "Refresh the page to reload data",
                    "Check if you have any data to load",
                    "Try the data recovery option in settings"
                ]
            elif "goal" in message.lower():
                suggestions = [
                    "Ensure your goal has realistic targets",
                    "Check that the timeframe is between 4-52 weeks",
                    "Verify weight values are reasonable"
                ]
            elif "progress" in message.lower():
                suggestions = [
                    "Check that weight values are realistic",
                    "Ensure the date is not in the future",
                    "Try entering fewer measurements at once"
                ]
            
            if suggestions:
                with st.expander("💡 Suggestions to fix this issue"):
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
        
        except Exception as e:
            logger.error(f"Error showing recovery suggestions: {str(e)}")
    
    def clear_messages(self):
        """Clear all stored feedback messages."""
        try:
            st.session_state.ui_feedback_messages = []
            logger.info("Cleared all feedback messages")
        except Exception as e:
            logger.error(f"Error clearing messages: {str(e)}")


# Global feedback manager instance
_feedback_manager = None


def get_feedback_manager() -> UIFeedbackManager:
    """Get the global feedback manager instance."""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = UIFeedbackManager()
    return _feedback_manager


# Convenience functions for common feedback operations
def show_success(message: str, details: Optional[str] = None):
    """Show a success message."""
    get_feedback_manager().show_success(message, details)


def show_error(message: str, error: Optional[Exception] = None, 
               show_details: bool = False, recoverable: bool = True):
    """Show an error message."""
    get_feedback_manager().show_error(message, error, show_details, recoverable)


def show_warning(message: str, details: Optional[str] = None):
    """Show a warning message."""
    get_feedback_manager().show_warning(message, details)


def show_info(message: str, details: Optional[str] = None):
    """Show an info message."""
    get_feedback_manager().show_info(message, details)


def show_validation_error(field_name: str, error_message: str, 
                         suggestions: Optional[list] = None):
    """Show a validation error."""
    get_feedback_manager().show_validation_error(field_name, error_message, suggestions)


def loading_indicator(message: str, key: Optional[str] = None):
    """Context manager for loading indicators."""
    return get_feedback_manager().loading_indicator(message, key)


def confirm_action(message: str, action_name: str, danger: bool = False) -> bool:
    """Show confirmation dialog."""
    return get_feedback_manager().confirm_action(message, action_name, danger)


# Decorator for handling errors in UI functions
def handle_ui_errors(operation_name: str, show_details: bool = False):
    """
    Decorator to handle errors in UI functions with user feedback.
    
    Args:
        operation_name: Name of the operation for error messages
        show_details: Whether to show technical error details
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = f"Failed to {operation_name.lower()}"
                show_error(error_message, e, show_details, recoverable=True)
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper
    return decorator


# Context manager for safe operations with user feedback
@contextmanager
def safe_operation(operation_name: str, loading_message: Optional[str] = None,
                  success_message: Optional[str] = None):
    """
    Context manager for safe operations with comprehensive error handling.
    
    Args:
        operation_name: Name of the operation
        loading_message: Optional loading message
        success_message: Optional success message
    """
    feedback = get_feedback_manager()
    
    try:
        # Show loading indicator if requested
        if loading_message:
            with feedback.loading_indicator(loading_message):
                yield
        else:
            yield
        
        # Show success message if provided
        if success_message:
            feedback.show_success(success_message)
    
    except Exception as e:
        # Show error with recovery suggestions
        error_message = f"Failed to {operation_name.lower()}"
        feedback.show_error(error_message, e, show_details=False, recoverable=True)
        logger.error(f"Error in safe operation '{operation_name}': {str(e)}")
        raise


# Validation helper functions
def validate_weight_input(weight: Optional[float], field_name: str = "Weight") -> bool:
    """
    Validate weight input with user feedback.
    
    Args:
        weight: Weight value to validate
        field_name: Name of the field for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if weight is None:
        return True  # Optional field
    
    if not isinstance(weight, (int, float)):
        show_validation_error(field_name, "Must be a number", 
                            ["Enter a numeric value", "Use decimal point for precision"])
        return False
    
    if weight <= 0:
        show_validation_error(field_name, "Must be greater than 0", 
                            ["Enter a positive weight value"])
        return False
    
    if weight < 30 or weight > 300:
        show_validation_error(field_name, "Must be between 30 and 300 kg", 
                            ["Check if the value is realistic", 
                             "Ensure you're using kilograms"])
        return False
    
    return True


def validate_measurement_input(measurement: Optional[float], field_name: str) -> bool:
    """
    Validate body measurement input with user feedback.
    
    Args:
        measurement: Measurement value to validate
        field_name: Name of the field for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if measurement is None:
        return True  # Optional field
    
    if not isinstance(measurement, (int, float)):
        show_validation_error(field_name, "Must be a number", 
                            ["Enter a numeric value", "Use decimal point for precision"])
        return False
    
    if measurement <= 0:
        show_validation_error(field_name, "Must be greater than 0", 
                            ["Enter a positive measurement"])
        return False
    
    if measurement > 200:
        show_validation_error(field_name, "Measurement seems too large", 
                            ["Check if the value is in centimeters", 
                             "Verify the measurement is correct"])
        return False
    
    return True


def validate_timeframe_input(timeframe: Optional[int], field_name: str = "Timeframe") -> bool:
    """
    Validate timeframe input with user feedback.
    
    Args:
        timeframe: Timeframe value to validate
        field_name: Name of the field for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if timeframe is None:
        show_validation_error(field_name, "Timeframe is required", 
                            ["Select a timeframe between 4-52 weeks"])
        return False
    
    if not isinstance(timeframe, int):
        show_validation_error(field_name, "Must be a whole number", 
                            ["Select from the slider or enter an integer"])
        return False
    
    if timeframe < 4 or timeframe > 52:
        show_validation_error(field_name, "Must be between 4 and 52 weeks", 
                            ["Choose a realistic timeframe", 
                             "Consider 8-24 weeks for most goals"])
        return False
    
    return True