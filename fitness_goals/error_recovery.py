"""
Error recovery and data validation utilities.

This module provides comprehensive error recovery mechanisms,
data validation, and system health checks for the fitness goals application.
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import json
import traceback

from .ui_feedback import show_success, show_error, show_warning, show_info

# Configure logging
logger = logging.getLogger(__name__)


class ErrorRecoveryManager:
    """
    Manages error recovery operations and system health checks.
    """
    
    def __init__(self):
        """Initialize the error recovery manager."""
        self.recovery_log = []
    
    def perform_system_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check session state integrity
            session_check = self._check_session_state_integrity()
            health_status['checks']['session_state'] = session_check
            
            # Check data storage integrity
            storage_check = self._check_data_storage_integrity()
            health_status['checks']['data_storage'] = storage_check
            
            # Check goal manager functionality
            goal_check = self._check_goal_manager_functionality()
            health_status['checks']['goal_manager'] = goal_check
            
            # Check progress tracker functionality
            progress_check = self._check_progress_tracker_functionality()
            health_status['checks']['progress_tracker'] = progress_check
            
            # Determine overall status
            failed_checks = [name for name, result in health_status['checks'].items() 
                           if not result.get('status', False)]
            
            if failed_checks:
                health_status['overall_status'] = 'degraded' if len(failed_checks) <= 2 else 'critical'
                health_status['errors'].extend(failed_checks)
            
            # Generate recommendations
            health_status['recommendations'] = self._generate_health_recommendations(health_status)
            
        except Exception as e:
            logger.error(f"Error during system health check: {str(e)}")
            health_status['overall_status'] = 'critical'
            health_status['errors'].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    def _check_session_state_integrity(self) -> Dict[str, Any]:
        """Check session state integrity."""
        result = {
            'status': True,
            'message': 'Session state is healthy',
            'details': {}
        }
        
        try:
            # Check required session state keys
            required_keys = ['user_id', 'data_storage_initialized']
            missing_keys = [key for key in required_keys if key not in st.session_state]
            
            if missing_keys:
                result['status'] = False
                result['message'] = f"Missing session state keys: {missing_keys}"
                result['details']['missing_keys'] = missing_keys
            
            # Check for corrupted data structures
            if 'fitness_goals' in st.session_state:
                goals = st.session_state.fitness_goals
                if not isinstance(goals, dict):
                    result['status'] = False
                    result['message'] = "Corrupted fitness goals data structure"
            
            if 'progress_history' in st.session_state:
                history = st.session_state.progress_history
                if not isinstance(history, dict):
                    result['status'] = False
                    result['message'] = "Corrupted progress history data structure"
            
            # Check session state size (memory usage)
            total_keys = len(st.session_state.keys())
            if total_keys > 100:
                result['details']['warning'] = f"Large number of session keys: {total_keys}"
            
            result['details']['total_keys'] = total_keys
            
        except Exception as e:
            result['status'] = False
            result['message'] = f"Session state check failed: {str(e)}"
        
        return result
    
    def _check_data_storage_integrity(self) -> Dict[str, Any]:
        """Check data storage integrity."""
        result = {
            'status': True,
            'message': 'Data storage is healthy',
            'details': {}
        }
        
        try:
            from .data_storage import get_data_storage
            
            storage = get_data_storage()
            user_id = storage._get_user_id()
            
            # Test basic storage operations
            stats = storage.get_storage_stats()
            result['details']['stats'] = stats
            
            # Validate data integrity
            integrity_results = storage.validate_data_integrity(user_id)
            result['details']['integrity'] = integrity_results
            
            if integrity_results.get('goal_error') or integrity_results.get('progress_errors'):
                result['status'] = False
                result['message'] = "Data integrity issues detected"
            
        except Exception as e:
            result['status'] = False
            result['message'] = f"Data storage check failed: {str(e)}"
        
        return result
    
    def _check_goal_manager_functionality(self) -> Dict[str, Any]:
        """Check goal manager functionality."""
        result = {
            'status': True,
            'message': 'Goal manager is functional',
            'details': {}
        }
        
        try:
            from .goal_manager import get_goal_manager
            
            goal_manager = get_goal_manager()
            user_id = st.session_state.get('user_id', 'default_user')
            
            # Test goal loading
            goal = goal_manager.get_user_goal(user_id)
            result['details']['has_goal'] = goal is not None
            
            if goal:
                # Test goal summary generation
                summary = goal_manager.get_goal_summary(user_id)
                result['details']['summary_available'] = summary is not None
            
        except Exception as e:
            result['status'] = False
            result['message'] = f"Goal manager check failed: {str(e)}"
        
        return result
    
    def _check_progress_tracker_functionality(self) -> Dict[str, Any]:
        """Check progress tracker functionality."""
        result = {
            'status': True,
            'message': 'Progress tracker is functional',
            'details': {}
        }
        
        try:
            from .progress_tracker import get_progress_tracker
            
            progress_tracker = get_progress_tracker()
            user_id = st.session_state.get('user_id', 'default_user')
            
            # Test progress loading
            history = progress_tracker.get_progress_history(user_id)
            result['details']['history_count'] = len(history)
            
            # Test summary generation
            summary = progress_tracker.get_progress_summary(user_id)
            result['details']['summary_available'] = summary is not None
            
        except Exception as e:
            result['status'] = False
            result['message'] = f"Progress tracker check failed: {str(e)}"
        
        return result
    
    def _generate_health_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on check results."""
        recommendations = []
        
        # Session state recommendations
        session_check = health_status['checks'].get('session_state', {})
        if not session_check.get('status', True):
            recommendations.append("Refresh the page to reset session state")
        
        # Data storage recommendations
        storage_check = health_status['checks'].get('data_storage', {})
        if not storage_check.get('status', True):
            recommendations.append("Use data recovery tools to fix corrupted data")
            recommendations.append("Export your data as backup before attempting repairs")
        
        # Goal manager recommendations
        goal_check = health_status['checks'].get('goal_manager', {})
        if not goal_check.get('status', True):
            recommendations.append("Try setting a new goal to reset goal manager")
        
        # Progress tracker recommendations
        progress_check = health_status['checks'].get('progress_tracker', {})
        if not progress_check.get('status', True):
            recommendations.append("Clear progress history if corruption is detected")
        
        # General recommendations
        if health_status['overall_status'] == 'critical':
            recommendations.append("Consider clearing all data and starting fresh")
            recommendations.append("Contact support if issues persist")
        
        return recommendations
    
    def attempt_automatic_recovery(self) -> Tuple[bool, str, List[str]]:
        """
        Attempt automatic recovery from common issues.
        
        Returns:
            Tuple of (success, message, actions_taken)
        """
        actions_taken = []
        
        try:
            # Perform health check first
            health_status = self.perform_system_health_check()
            
            if health_status['overall_status'] == 'healthy':
                return True, "System is healthy, no recovery needed", []
            
            # Attempt session state recovery
            if 'session_state' in health_status['errors']:
                success = self._recover_session_state()
                if success:
                    actions_taken.append("Reset corrupted session state")
            
            # Attempt data storage recovery
            if 'data_storage' in health_status['errors']:
                success = self._recover_data_storage()
                if success:
                    actions_taken.append("Repaired data storage issues")
            
            # Re-check health after recovery attempts
            new_health_status = self.perform_system_health_check()
            
            if new_health_status['overall_status'] == 'healthy':
                return True, "Automatic recovery successful", actions_taken
            elif len(actions_taken) > 0:
                return True, "Partial recovery completed", actions_taken
            else:
                return False, "Automatic recovery failed", actions_taken
        
        except Exception as e:
            logger.error(f"Error during automatic recovery: {str(e)}")
            return False, f"Recovery failed: {str(e)}", actions_taken
    
    def _recover_session_state(self) -> bool:
        """Attempt to recover corrupted session state."""
        try:
            # Reset essential session state keys
            if 'user_id' not in st.session_state:
                st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if 'data_storage_initialized' not in st.session_state:
                st.session_state.data_storage_initialized = True
            
            # Initialize data structures if corrupted
            if 'fitness_goals' not in st.session_state or not isinstance(st.session_state.fitness_goals, dict):
                st.session_state.fitness_goals = {}
            
            if 'progress_history' not in st.session_state or not isinstance(st.session_state.progress_history, dict):
                st.session_state.progress_history = {}
            
            return True
        
        except Exception as e:
            logger.error(f"Error recovering session state: {str(e)}")
            return False
    
    def _recover_data_storage(self) -> bool:
        """Attempt to recover data storage issues."""
        try:
            from .data_storage import get_data_storage
            
            storage = get_data_storage()
            user_id = storage._get_user_id()
            
            # Attempt to recover from browser storage
            storage._restore_from_browser_storage()
            
            # Validate recovered data
            integrity_results = storage.validate_data_integrity(user_id)
            
            return not (integrity_results.get('goal_error') or integrity_results.get('progress_errors'))
        
        except Exception as e:
            logger.error(f"Error recovering data storage: {str(e)}")
            return False
    
    def create_diagnostic_report(self) -> Dict[str, Any]:
        """Create comprehensive diagnostic report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_check': self.perform_system_health_check(),
            'session_info': self._get_session_info(),
            'error_log': self.recovery_log[-10:],  # Last 10 errors
            'system_info': self._get_system_info()
        }
        
        return report
    
    def _get_session_info(self) -> Dict[str, Any]:
        """Get session information for diagnostics."""
        try:
            return {
                'user_id': st.session_state.get('user_id', 'unknown'),
                'session_keys': list(st.session_state.keys()),
                'session_key_count': len(st.session_state.keys()),
                'has_goal': 'user_goal' in st.session_state and st.session_state.user_goal is not None,
                'message_count': len(st.session_state.get('messages', [])),
                'data_initialized': st.session_state.get('data_storage_initialized', False)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        try:
            return {
                'streamlit_version': st.__version__,
                'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                'current_time': datetime.now().isoformat(),
                'timezone': str(datetime.now().astimezone().tzinfo)
            }
        except Exception as e:
            return {'error': str(e)}


# Global error recovery manager instance
_recovery_manager = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


def perform_health_check() -> Dict[str, Any]:
    """Perform system health check."""
    return get_recovery_manager().perform_system_health_check()


def attempt_recovery() -> Tuple[bool, str, List[str]]:
    """Attempt automatic system recovery."""
    return get_recovery_manager().attempt_automatic_recovery()


def create_diagnostic_report() -> Dict[str, Any]:
    """Create diagnostic report."""
    return get_recovery_manager().create_diagnostic_report()


def render_system_diagnostics():
    """Render system diagnostics interface."""
    st.subheader("🔍 System Diagnostics")
    
    # Health check section
    if st.button("🏥 Run Health Check", key="health_check_btn"):
        with st.spinner("Running system health check..."):
            health_status = perform_health_check()
        
        # Display results
        if health_status['overall_status'] == 'healthy':
            show_success("System is healthy! 🎉", "All components are functioning normally.")
        elif health_status['overall_status'] == 'degraded':
            show_warning("System has minor issues", "Some components may not be functioning optimally.")
        else:
            show_error("System has critical issues", None, recoverable=True)
        
        # Show detailed results
        with st.expander("📋 Detailed Health Check Results"):
            st.json(health_status)
    
    # Recovery section
    st.subheader("🔧 Automatic Recovery")
    
    if st.button("🚑 Attempt Recovery", key="recovery_btn"):
        with st.spinner("Attempting automatic recovery..."):
            success, message, actions = attempt_recovery()
        
        if success:
            show_success("Recovery completed!", message)
            if actions:
                show_info("Actions taken:", "\n".join(f"• {action}" for action in actions))
        else:
            show_error("Recovery failed", None, recoverable=True)
            show_info("You may need to manually clear data or refresh the page.")
    
    # Diagnostic report section
    st.subheader("📊 Diagnostic Report")
    
    if st.button("📋 Generate Report", key="diagnostic_btn"):
        with st.spinner("Generating diagnostic report..."):
            report = create_diagnostic_report()
        
        # Display summary
        show_success("Diagnostic report generated!")
        
        # Download button
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="💾 Download Report",
            data=report_json,
            file_name=f"fitness_app_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Show preview
        with st.expander("🔍 Report Preview"):
            st.json(report)