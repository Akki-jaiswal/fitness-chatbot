"""
User interface components for data management features.

This module provides Streamlit UI components for data backup, restore,
clearing, and error recovery functionality.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from .data_storage import get_data_storage, DataStorageError
from .models import FitnessGoal, ProgressEntry
from .ui_feedback import (
    show_success, show_error, show_warning, show_info, show_validation_error,
    loading_indicator, confirm_action, safe_operation, handle_ui_errors
)

# Configure logging
logger = logging.getLogger(__name__)


@handle_ui_errors("render data management interface")
def render_data_management_ui():
    """
    Render the complete data management interface.
    """
    st.header("🔧 Data Management")
    
    # Get current user ID with error handling
    try:
        storage = get_data_storage()
        current_user_id = storage._get_user_id()
    except Exception as e:
        show_error("Failed to initialize data management", e, recoverable=True)
        return
    
    # Create tabs for different data management functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "💾 Backup & Export", 
        "📥 Import & Restore", 
        "🔄 Recovery & Cleanup"
    ])
    
    with tab1:
        render_data_overview(storage, current_user_id)
    
    with tab2:
        render_backup_export_ui(storage, current_user_id)
    
    with tab3:
        render_import_restore_ui(storage, current_user_id)
    
    with tab4:
        render_recovery_cleanup_ui(storage, current_user_id)
    
    # Add system diagnostics section
    st.divider()
    from .error_recovery import render_system_diagnostics
    render_system_diagnostics()


def render_data_overview(storage, user_id: str):
    """
    Render data overview and statistics.
    """
    st.subheader("Data Overview")
    
    try:
        # Get storage statistics
        stats = storage.get_storage_stats()
        
        # Display current user info
        st.info(f"**Current User ID:** `{user_id}`")
        
        # Display statistics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Goals", stats.get('total_goals', 0))
        
        with col2:
            st.metric("Progress Entries", stats.get('total_progress_entries', 0))
        
        with col3:
            st.metric("Users with Data", stats.get('users_with_progress', 0))
        
        # Data integrity check
        st.subheader("Data Integrity")
        
        if st.button("🔍 Check Data Integrity", key="check_integrity"):
            with st.spinner("Checking data integrity..."):
                integrity_results = storage.validate_data_integrity(user_id)
                
                # Display results
                if integrity_results['goal_valid']:
                    st.success("✅ Goal data is valid")
                elif integrity_results['goal_error']:
                    st.error(f"❌ Goal data error: {integrity_results['goal_error']}")
                else:
                    st.info("ℹ️ No goal data found")
                
                if integrity_results['progress_entries_valid'] > 0:
                    st.success(f"✅ {integrity_results['progress_entries_valid']} valid progress entries")
                
                if integrity_results['progress_errors']:
                    for error in integrity_results['progress_errors']:
                        st.error(f"❌ Progress data error: {error}")
                
                if not integrity_results['goal_valid'] and integrity_results['progress_entries_valid'] == 0:
                    st.info("ℹ️ No data found for current user")
        
        # Display current data summary
        st.subheader("Current Data Summary")
        
        try:
            # Show goal information
            goal = storage.load_goal(user_id)
            if goal:
                st.success("**Current Goal:**")
                st.write(f"- Type: {goal.goal_type.replace('_', ' ').title()}")
                st.write(f"- Current: {goal.current_value} kg")
                st.write(f"- Target: {goal.target_value} kg")
                st.write(f"- Deadline: {goal.target_date.strftime('%Y-%m-%d')}")
                st.write(f"- Days remaining: {goal.days_remaining()}")
            else:
                st.info("No goal set")
            
            # Show progress summary
            progress_history = storage.load_progress_history(user_id)
            if progress_history:
                st.success(f"**Progress History:** {len(progress_history)} entries")
                if len(progress_history) > 0:
                    latest_entry = progress_history[0]  # Most recent first
                    st.write(f"- Latest entry: {latest_entry.date.strftime('%Y-%m-%d')}")
                    if latest_entry.weight:
                        st.write(f"- Latest weight: {latest_entry.weight} kg")
            else:
                st.info("No progress entries")
                
        except Exception as e:
            st.error(f"Error loading data summary: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading data overview: {str(e)}")


@handle_ui_errors("render backup and export interface")
def render_backup_export_ui(storage, user_id: str):
    """
    Render backup and export interface with enhanced error handling.
    """
    st.subheader("Backup & Export")
    
    st.write("Create backups of your fitness data for safekeeping or transfer to another device.")
    
    # Export data as JSON
    st.subheader("📄 Export Data (JSON)")
    
    if st.button("📤 Export My Data", key="export_json"):
        try:
            with safe_operation("export data", 
                              loading_message="Preparing your data for export...",
                              success_message="Data exported successfully! 🎉"):
                
                export_data = storage.export_user_data(user_id)
                
                if export_data:
                    # Display export summary
                    col1, col2 = st.columns(2)
                    with col1:
                        has_goal = "✅" if export_data['goal'] else "❌"
                        st.write(f"{has_goal} Goal data")
                    with col2:
                        entry_count = len(export_data['progress_history'])
                        st.write(f"📊 {entry_count} progress entries")
                    
                    # Provide download button
                    json_data = json.dumps(export_data, indent=2, default=str)
                    filename = f"fitness_data_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    st.download_button(
                        label="💾 Download JSON File",
                        data=json_data,
                        file_name=filename,
                        mime="application/json",
                        key="download_json"
                    )
                    
                    show_info(f"Export includes: {'goal data, ' if export_data['goal'] else ''}{len(export_data['progress_history'])} progress entries")
                    
                    # Show preview of exported data
                    with st.expander("🔍 Preview Exported Data"):
                        st.json(export_data)
                
                else:
                    show_warning("No data found to export", 
                               "Set a goal and log some progress first, then try exporting again.")
                    
        except DataStorageError as e:
            show_error("Failed to export data", e, recoverable=True)
        except Exception as e:
            show_error("Unexpected error during export", e, recoverable=True)
    
    # Create backup file (ZIP)
    st.subheader("📦 Create Backup File")
    
    if st.button("📦 Create Backup ZIP", key="create_backup"):
        try:
            with safe_operation("create backup file", 
                              loading_message="Creating backup file...",
                              success_message="Backup file created successfully! 🎉"):
                
                backup_bytes = storage.create_backup_file(user_id)
                
                if backup_bytes:
                    filename = f"fitness_backup_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    
                    st.download_button(
                        label="💾 Download Backup ZIP",
                        data=backup_bytes,
                        file_name=filename,
                        mime="application/zip",
                        key="download_backup"
                    )
                    
                    show_info("The backup ZIP contains your data in JSON format plus metadata and instructions.")
                
                else:
                    show_warning("No data found to backup", 
                               "Set a goal and log some progress first, then try creating a backup.")
                    
        except DataStorageError as e:
            show_error("Failed to create backup", e, recoverable=True)
        except Exception as e:
            show_error("Unexpected error during backup creation", e, recoverable=True)


def render_import_restore_ui(storage, user_id: str):
    """
    Render import and restore interface.
    """
    st.subheader("Import & Restore")
    
    st.write("Import data from backups or transfer data from another device.")
    
    # Import from JSON file
    st.subheader("📄 Import from JSON")
    
    uploaded_json = st.file_uploader(
        "Choose a JSON file to import",
        type=['json'],
        key="upload_json",
        help="Upload a JSON file exported from the fitness goals app"
    )
    
    if uploaded_json is not None:
        try:
            # Read and parse JSON
            json_content = uploaded_json.read().decode('utf-8')
            import_data = json.loads(json_content)
            
            # Display import preview
            st.success("✅ JSON file loaded successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                has_goal = "✅" if import_data.get('goal') else "❌"
                st.write(f"{has_goal} Goal data")
            with col2:
                entry_count = len(import_data.get('progress_history', []))
                st.write(f"📊 {entry_count} progress entries")
            
            # Show import options
            overwrite = st.checkbox(
                "Overwrite existing data",
                key="overwrite_json",
                help="Check this to replace your current data with the imported data"
            )
            
            if st.button("📥 Import Data", key="import_json_btn"):
                with st.spinner("Importing data..."):
                    success, message = storage.import_user_data(user_id, import_data, overwrite)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()  # Refresh the page to show updated data
                    else:
                        st.error(f"❌ {message}")
        
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file format")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Restore from backup ZIP
    st.subheader("📦 Restore from Backup")
    
    uploaded_zip = st.file_uploader(
        "Choose a backup ZIP file to restore",
        type=['zip'],
        key="upload_zip",
        help="Upload a ZIP backup file created by the fitness goals app"
    )
    
    if uploaded_zip is not None:
        try:
            # Read ZIP file
            zip_bytes = uploaded_zip.read()
            
            st.success("✅ Backup file loaded successfully!")
            
            # Show restore options
            overwrite_zip = st.checkbox(
                "Overwrite existing data",
                key="overwrite_zip",
                help="Check this to replace your current data with the backup data"
            )
            
            if st.button("📥 Restore from Backup", key="restore_zip_btn"):
                with st.spinner("Restoring from backup..."):
                    success, message = storage.restore_from_backup_file(user_id, zip_bytes, overwrite_zip)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()  # Refresh the page to show updated data
                    else:
                        st.error(f"❌ {message}")
        
        except Exception as e:
            st.error(f"❌ Error reading backup file: {str(e)}")
    
    # Restore from session backup
    if 'last_data_backup' in st.session_state:
        st.subheader("🔄 Restore from Recent Backup")
        
        backup_info = st.session_state['last_data_backup']
        backup_time = datetime.fromisoformat(backup_info['timestamp'])
        
        st.info(f"💾 Recent backup available from {backup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("🔄 Restore Recent Backup", key="restore_recent"):
            with st.spinner("Restoring recent backup..."):
                success, message = storage.import_user_data(user_id, backup_info['data'], overwrite=True)
                
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")


def render_recovery_cleanup_ui(storage, user_id: str):
    """
    Render recovery and cleanup interface.
    """
    st.subheader("Recovery & Cleanup")
    
    st.write("Recover from data corruption or clear all data with confirmation.")
    
    # Data recovery
    st.subheader("🔄 Data Recovery")
    
    st.write("Attempt to recover corrupted or missing data from backups and browser storage.")
    
    if st.button("🔄 Recover Data", key="recover_data"):
        with st.spinner("Attempting data recovery..."):
            try:
                recovery_results = storage.recover_corrupted_data(user_id)
                
                # Display recovery results
                if recovery_results['actions_taken']:
                    st.success("✅ Recovery actions completed:")
                    for action in recovery_results['actions_taken']:
                        st.write(f"- {action}")
                
                if recovery_results['recovered_items']:
                    st.success("✅ Recovered items:")
                    for item in recovery_results['recovered_items']:
                        st.write(f"- {item}")
                
                if recovery_results['warnings']:
                    st.warning("⚠️ Warnings:")
                    for warning in recovery_results['warnings']:
                        st.write(f"- {warning}")
                
                if recovery_results['unrecoverable_items']:
                    st.error("❌ Unrecoverable items:")
                    for item in recovery_results['unrecoverable_items']:
                        st.write(f"- {item}")
                
                if not any([
                    recovery_results['actions_taken'],
                    recovery_results['recovered_items'],
                    recovery_results['warnings'],
                    recovery_results['unrecoverable_items']
                ]):
                    st.info("ℹ️ No recovery actions needed - data appears to be intact")
                
            except Exception as e:
                st.error(f"❌ Recovery failed: {str(e)}")
    
    # Session restoration
    st.subheader("💾 Session Restoration")
    
    st.write("Restore data from browser local storage.")
    
    if st.button("💾 Restore from Browser Storage", key="restore_session"):
        with st.spinner("Restoring session data..."):
            try:
                success, message = storage.restore_session_from_browser_storage()
                
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.warning(f"⚠️ {message}")
                    
            except Exception as e:
                st.error(f"❌ Session restoration failed: {str(e)}")
    
    # Data clearing with confirmation
    st.subheader("🗑️ Clear All Data")
    
    st.warning("⚠️ **Warning:** This will permanently delete all your fitness goals and progress data!")
    
    # Two-step confirmation process
    if 'confirm_clear_step1' not in st.session_state:
        st.session_state.confirm_clear_step1 = False
    
    if not st.session_state.confirm_clear_step1:
        if st.button("🗑️ I want to clear my data", key="clear_step1"):
            st.session_state.confirm_clear_step1 = True
            st.rerun()
    else:
        st.error("⚠️ **Final confirmation required!**")
        st.write("This action cannot be undone. All your data will be permanently deleted.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❌ Cancel", key="clear_cancel"):
                st.session_state.confirm_clear_step1 = False
                st.rerun()
        
        with col2:
            if st.button("🗑️ Yes, delete everything", key="clear_confirm", type="primary"):
                with st.spinner("Clearing all data..."):
                    try:
                        success, message = storage.clear_user_data_with_confirmation(user_id)
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.confirm_clear_step1 = False
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            
                    except Exception as e:
                        st.error(f"❌ Clear operation failed: {str(e)}")


def render_data_management_sidebar():
    """
    Render a compact data management widget for the sidebar.
    """
    with st.expander("🔧 Data Management"):
        storage = get_data_storage()
        user_id = storage._get_user_id()
        
        # Quick stats
        try:
            stats = storage.get_storage_stats()
            st.write(f"Goals: {stats.get('total_goals', 0)}")
            st.write(f"Progress entries: {stats.get('total_progress_entries', 0)}")
        except:
            st.write("Stats unavailable")
        
        # Quick actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export", key="sidebar_export", help="Export your data"):
                try:
                    export_data = storage.export_user_data(user_id)
                    if export_data:
                        json_data = json.dumps(export_data, indent=2, default=str)
                        filename = f"fitness_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        st.download_button(
                            "💾 Download",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            key="sidebar_download"
                        )
                    else:
                        st.warning("No data to export")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            if st.button("🔄 Recover", key="sidebar_recover", help="Recover corrupted data"):
                try:
                    recovery_results = storage.recover_corrupted_data(user_id)
                    if recovery_results['recovered_items']:
                        st.success("Data recovered!")
                    else:
                        st.info("No recovery needed")
                except Exception as e:
                    st.error(f"Recovery failed: {str(e)}")