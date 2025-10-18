"""
User onboarding flow for first-time goal setting.
Provides a guided experience for new users to set up their fitness goals.
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .models import FitnessGoal
from .goal_manager import get_goal_manager
from .ui_feedback import show_success, show_info, show_warning, handle_ui_errors

logger = logging.getLogger(__name__)


@handle_ui_errors("onboarding flow")
def should_show_onboarding() -> bool:
    """
    Determine if the onboarding flow should be shown to the user.
    
    Returns:
        bool: True if onboarding should be shown, False otherwise
    """
    try:
        # Check if user has completed onboarding
        if st.session_state.get('onboarding_completed', False):
            return False
        
        # Check if user has an existing goal
        user_id = st.session_state.get('user_id', 'default_user')
        goal_manager = get_goal_manager()
        user_goal = goal_manager.get_user_goal(user_id)
        
        if user_goal and user_goal.is_active:
            # User has an active goal, mark onboarding as completed
            st.session_state.onboarding_completed = True
            return False
        
        # Check if user has dismissed onboarding recently
        last_dismissed = st.session_state.get('onboarding_dismissed_at')
        if last_dismissed:
            # Don't show again for 24 hours after dismissal
            if datetime.now() - last_dismissed < timedelta(hours=24):
                return False
        
        # Check conversation count to avoid showing too early
        conversation_count = len(st.session_state.get('messages', []))
        
        # Show onboarding after user has had a few interactions but no goal set
        return conversation_count >= 2
    
    except Exception as e:
        logger.error(f"Error checking onboarding status: {str(e)}")
        return False


@handle_ui_errors("onboarding welcome")
def show_onboarding_welcome() -> None:
    """Display the welcome message and onboarding introduction."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="margin: 0 0 1rem 0;">🎯 Welcome to Your Fitness Journey!</h2>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
            I notice you haven't set up a fitness goal yet. Let me help you create a personalized 
            plan to track your progress and stay motivated!
        </p>
    </div>
    """, unsafe_allow_html=True)


@handle_ui_errors("onboarding goal setup")
def render_onboarding_goal_setup() -> Optional[FitnessGoal]:
    """
    Render the guided goal setup form for onboarding.
    
    Returns:
        Optional[FitnessGoal]: Created goal if successful, None otherwise
    """
    st.markdown("### 🚀 Let's Set Up Your First Goal")
    
    # Step 1: Goal type selection with descriptions
    st.markdown("**Step 1: What's your main fitness objective?**")
    
    goal_descriptions = {
        "weight_loss": {
            "title": "🔥 Weight Loss",
            "description": "Lose weight through a combination of cardio, strength training, and nutrition guidance",
            "benefits": ["Burn calories efficiently", "Improve cardiovascular health", "Build lean muscle"]
        },
        "muscle_building": {
            "title": "💪 Muscle Building", 
            "description": "Build muscle mass and strength through progressive resistance training",
            "benefits": ["Increase muscle mass", "Boost metabolism", "Improve functional strength"]
        },
        "weight_maintenance": {
            "title": "⚖️ Weight Maintenance",
            "description": "Maintain current weight while improving fitness and body composition",
            "benefits": ["Maintain healthy weight", "Improve fitness level", "Build sustainable habits"]
        }
    }
    
    # Display goal options with enhanced descriptions
    selected_goal_type = None
    cols = st.columns(3)
    
    for i, (goal_type, info) in enumerate(goal_descriptions.items()):
        with cols[i]:
            if st.button(
                f"{info['title']}\n\n{info['description']}", 
                key=f"onboarding_goal_{goal_type}",
                help=f"Benefits: {', '.join(info['benefits'])}"
            ):
                selected_goal_type = goal_type
                st.session_state.onboarding_selected_goal_type = goal_type
    
    # Use previously selected goal type if available
    if not selected_goal_type:
        selected_goal_type = st.session_state.get('onboarding_selected_goal_type')
    
    if not selected_goal_type:
        st.info("👆 Please select your fitness goal to continue")
        return None
    
    # Show selected goal confirmation
    selected_info = goal_descriptions[selected_goal_type]
    st.success(f"Great choice! You selected: **{selected_info['title']}**")
    
    # Step 2: Current and target values
    st.markdown("**Step 2: Tell me about your current situation**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_weight = st.number_input(
            "Current Weight (kg)",
            min_value=30.0,
            max_value=300.0,
            value=70.0,
            step=0.1,
            help="Enter your current weight in kilograms"
        )
    
    with col2:
        if selected_goal_type == "weight_loss":
            target_weight = st.number_input(
                "Target Weight (kg)",
                min_value=30.0,
                max_value=current_weight - 0.5,
                value=max(30.0, current_weight - 5.0),
                step=0.1,
                help="Your goal weight (must be less than current weight)"
            )
        elif selected_goal_type == "muscle_building":
            target_weight = st.number_input(
                "Target Weight (kg)",
                min_value=current_weight + 0.5,
                max_value=300.0,
                value=current_weight + 3.0,
                step=0.1,
                help="Your goal weight (muscle gain typically 0.5-1kg per month)"
            )
        else:  # weight_maintenance
            target_weight = st.number_input(
                "Target Weight Range (kg)",
                min_value=current_weight - 2.0,
                max_value=current_weight + 2.0,
                value=current_weight,
                step=0.1,
                help="Maintain within 1-2kg of current weight"
            )
    
    # Step 3: Timeframe selection with guidance
    st.markdown("**Step 3: What's your timeline?**")
    
    # Provide realistic timeframe suggestions based on goal type
    if selected_goal_type == "weight_loss":
        weight_diff = current_weight - target_weight
        recommended_weeks = max(8, int(weight_diff * 2))  # ~0.5kg per week
        st.info(f"💡 For healthy weight loss of {weight_diff:.1f}kg, we recommend at least {recommended_weeks} weeks")
    elif selected_goal_type == "muscle_building":
        weight_diff = target_weight - current_weight
        recommended_weeks = max(12, int(weight_diff * 4))  # ~0.25kg per week
        st.info(f"💡 For muscle gain of {weight_diff:.1f}kg, we recommend at least {recommended_weeks} weeks")
    else:
        recommended_weeks = 12
        st.info("💡 For weight maintenance goals, we recommend starting with 12 weeks to build habits")
    
    timeframe_weeks = st.slider(
        "Timeline (weeks)",
        min_value=4,
        max_value=52,
        value=min(recommended_weeks, 24),
        step=1,
        help="Choose a realistic timeframe for your goal"
    )
    
    # Step 4: Goal summary and confirmation
    st.markdown("**Step 4: Review Your Goal**")
    
    target_date = datetime.now() + timedelta(weeks=timeframe_weeks)
    
    st.markdown(f"""
    <div style="
        background-color: #f8f9fa;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    ">
        <h4 style="color: #667eea; margin-top: 0;">📋 Your Goal Summary</h4>
        <ul style="margin-bottom: 0;">
            <li><strong>Goal Type:</strong> {selected_info['title']}</li>
            <li><strong>Current Weight:</strong> {current_weight}kg</li>
            <li><strong>Target Weight:</strong> {target_weight}kg</li>
            <li><strong>Timeline:</strong> {timeframe_weeks} weeks</li>
            <li><strong>Target Date:</strong> {target_date.strftime('%B %d, %Y')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create and save goal
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎯 Create My Goal", type="primary", use_container_width=True):
            try:
                goal_manager = get_goal_manager()
                user_id = st.session_state.get('user_id', 'default_user')
                
                # Create goal data
                goal_data = {
                    'goal_type': selected_goal_type,
                    'current_value': current_weight,
                    'target_value': target_weight,
                    'timeframe_weeks': timeframe_weeks
                }
                
                # Validate and create goal
                is_valid, error_message = goal_manager.validate_goal_data(goal_data)
                if not is_valid:
                    show_warning(f"Goal validation failed: {error_message}")
                    return None
                
                # Create the goal
                new_goal = goal_manager.create_goal(goal_data)
                
                if new_goal:
                    # Mark onboarding as completed
                    st.session_state.onboarding_completed = True
                    st.session_state.user_goal = new_goal
                    
                    # Clear onboarding state
                    if 'onboarding_selected_goal_type' in st.session_state:
                        del st.session_state.onboarding_selected_goal_type
                    
                    show_success(
                        "🎉 Goal Created Successfully!",
                        f"Your {selected_info['title'].lower()} goal has been set up. "
                        "You can now track your progress and get personalized recommendations!"
                    )
                    
                    # Show next steps
                    st.markdown("""
                    ### 🚀 What's Next?
                    
                    1. **📊 Track Progress**: Use the Progress Tracking tab to log your weight and measurements
                    2. **💬 Get Advice**: Ask me questions about workouts and nutrition - I'll personalize responses to your goal
                    3. **🎯 Stay Motivated**: I'll help you track milestones and celebrate your achievements
                    
                    Ready to start your fitness journey? Ask me anything!
                    """)
                    
                    return new_goal
                else:
                    show_warning("Failed to create goal. Please try again.")
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating goal during onboarding: {str(e)}")
                show_warning(f"Error creating goal: {str(e)}")
                return None
    
    return None


@handle_ui_errors("onboarding dismissal")
def handle_onboarding_dismissal() -> None:
    """Handle user dismissing the onboarding flow."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("⏭️ Skip Goal Setup (I'll set it up later)", use_container_width=True):
            st.session_state.onboarding_dismissed_at = datetime.now()
            show_info(
                "Onboarding skipped",
                "You can set up your fitness goal anytime using the sidebar. "
                "I'll remind you again tomorrow!"
            )
            st.rerun()


@handle_ui_errors("onboarding flow")
def render_onboarding_flow() -> Optional[FitnessGoal]:
    """
    Render the complete onboarding flow.
    
    Returns:
        Optional[FitnessGoal]: Created goal if user completes onboarding, None otherwise
    """
    if not should_show_onboarding():
        return None
    
    # Show onboarding in a container
    with st.container():
        show_onboarding_welcome()
        
        # Render goal setup
        created_goal = render_onboarding_goal_setup()
        
        if created_goal:
            return created_goal
        
        # Show dismissal option
        st.markdown("---")
        handle_onboarding_dismissal()
    
    return None


@handle_ui_errors("onboarding check")
def check_and_show_onboarding() -> Optional[FitnessGoal]:
    """
    Check if onboarding should be shown and render it if needed.
    This is the main function to call from the main app.
    
    Returns:
        Optional[FitnessGoal]: Created goal if user completes onboarding, None otherwise
    """
    try:
        if should_show_onboarding():
            st.markdown("---")
            return render_onboarding_flow()
        return None
    except Exception as e:
        logger.error(f"Error in onboarding check: {str(e)}")
        return None