import re
import streamlit as st
import logging
from fitness_goals.enhanced_chatbot import get_enhanced_chatbot_response
from fitness_goals.goal_manager import get_goal_manager
from fitness_goals.data_storage import get_data_storage
from fitness_goals.goal_ui import render_goal_setting_sidebar, initialize_goal_ui_session_state
from fitness_goals.progress_ui import render_progress_tracking_interface
from fitness_goals.ui_feedback import (
    show_success, show_error, show_warning, show_info,
    loading_indicator, safe_operation, handle_ui_errors
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Chatbot Rules Definition for a Fitness App ---
chatbot_rules = {
    # Greetings
    r".*\b(hi|hello|hey|greetings)\b.*": {
        'response': "Hello there! 👋 Welcome to your fitness companion. How can I help you today?",
        'intent': 'greeting'
    },
    r".*\b(how are you|how's it going)\b.*": {
        'response': "I'm a bot 🤖 here to help with your fitness journey! How are you feeling today?",
        'intent': 'greeting'
    },
    r".*\b(what is your name|who are you)\b.*": {
        'response': "I'm your friendly **Fitness Bot** 💪 Ask me about workouts, meals, or sleep!",
        'intent': 'greeting'
    },

    # Workout-related
    r".*\b(workout plan|exercise routine|gym plan|workout|exercise|gym)\b.*": {
        'response': "I can help with workout plans! 🏋️ Are you looking for beginners, strength, cardio, or flexibility?",
        'intent': 'workout_plan'
    },
    r".*\b(beginner|start exercising)\b.*": {
        'response': "For beginners, start with squats, push-ups, and planks. 🔥 Consistency is key!",
        'intent': 'beginner_workout'
    },
    r".*\b(strength|strength training|build muscle)\b.*": {
        'response': "Strength training tip: focus on squats, deadlifts, bench press, and overhead press. 🏋️",
        'intent': 'strength_workout'
    },
    r".*\b(cardio|endurance)\b.*": {
        'response': "Cardio keeps your heart strong ❤️ Try running, cycling, or swimming!",
        'intent': 'cardio_workout'
    },
    r".*\b(flexibility|stretching|yoga)\b.*": {
        'response': "Flexibility training 🧘 helps recovery. Try yoga or daily stretching for 10–15 mins.",
        'intent': 'flexibility'
    },
    r".*\b(warm up|cool down|warmup|cooldown)\b.*": {
        'response': "Always warm up for 5–10 mins before and cool down after workouts to avoid injuries. ✅",
        'intent': 'workout_prep'
    },
    r".*\b(how many times a week|workout frequency|frequency)\b.*": {
        'response': "Aim for 3–5 workout sessions per week 💡 and give your body time to rest.",
        'intent': 'workout_frequency'
    },

    # Nutrition
    r".*\b(healthy meals|diet plan|nutrition advice|meals|diet|nutrition)\b.*": {
        'response': "Nutrition is key! 🥗 Want ideas for breakfast, lunch, dinner, or snacks?",
        'intent': 'nutrition_plan'
    },
    r".*\b(breakfast ideas|healthy breakfast|breakfast)\b.*": {
        'response': "Try oatmeal with fruits, Greek yogurt with berries, or eggs with veggies. 🍳",
        'intent': 'breakfast_ideas'
    },
    r".*\b(lunch ideas|healthy lunch|lunch)\b.*": {
        'response': "Healthy lunch 🥗: grilled chicken with veggies, quinoa salad, or lentils with rice.",
        'intent': 'lunch_ideas'
    },
    r".*\b(dinner ideas|healthy dinner|dinner)\b.*": {
        'response': "For dinner 🍽️: salmon with sweet potatoes, veggie stir-fry, or whole-grain pasta.",
        'intent': 'dinner_ideas'
    },
    r".*\b(snack ideas|healthy snack|snacks)\b.*": {
        'response': "Snack smart! 🍏 Nuts, fruit, hummus with carrots, or yogurt with seeds.",
        'intent': 'snack_ideas'
    },
    r".*\b(meal prep|prepare food|mealprep)\b.*": {
        'response': "Meal prep tip: cook proteins, carbs, and veggies in bulk on weekends. 🍱",
        'intent': 'meal_prep'
    },
    r".*\b(calorie intake|how many calories|kg|kilogram|kgs|weight)\b.*": {
        'response': "Calorie needs vary. ⚖️ Best to consult a professional, but I can share general nutrition principles.",
        'intent': 'weight_calories'
    },
    r".*\b(protein|carbs|fats)\b.*": {
        'response': "Balanced meals: protein for repair, carbs for energy, fats for health. 🥩🍞🥑",
        'intent': 'macros'
    },

    # Sleep
    r".*\b(improve sleep|sleep better|sleep tips|sleep)\b.*": {
        'response': "Sleep well 😴 Keep a routine, reduce screens before bed, and rest 7–9 hrs.",
        'intent': 'sleep_tips'
    },
    r".*\b(how much sleep|hours of sleep)\b.*": {
        'response': "Most adults need 7–9 hours of good sleep per night. 🌙",
        'intent': 'sleep_duration'
    },
    r".*\b(insomnia|can't sleep)\b.*": {
        'response': "Try relaxation, avoid caffeine, and make your room sleep-friendly. 🛏️",
        'intent': 'insomnia_help'
    },

    # Motivation & Features
    r".*\b(track progress|monitor goals|track|progress|goals)\b.*": {
        'response': "📊 You can track workouts, meals, and sleep progress inside the app.",
        'intent': 'app_features'
    },
    r".*\b(app features|what can this app do|features)\b.*": {
        'response': "This app offers workout plans, meal tracking, sleep logs, and goal setting. 🚀",
        'intent': 'app_features'
    },
    r".*\b(motivation|stay motivated)\b.*": {
        'response': "💡 Motivation tip: set small goals, find a buddy, and celebrate wins!",
        'intent': 'motivation'
    },

    # Help
    r".*\b(help)\b.*": {
        'response': ("You can ask me about workouts, meals, sleep, and motivation. 🤖\n"
                     "Try typing: 'workout plan', 'healthy meals', 'sleep tips', or 'motivate me'."),
        'intent': 'help'
    },


    # Polite Closings
    r".*\b(thank you|thanks)\b.*": {
        'response': "You're welcome! 🙌 Keep pushing towards your goals!",
        'intent': 'thank_you'
    },
    r".*\b(bye|goodbye|exit|quit|see you)\b.*": {
        'response': "Goodbye 👋 Stay fit and healthy!",
        'intent': 'exit'
    },

    # Default
    "default": {
        'response': "🤔 I'm not sure about that. Type 'help' to see what I can do!",
        'intent': 'unknown'
    }
}

last_matched_intent = None


def clean_input(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)
    return user_input


@handle_ui_errors("generate chatbot response")
def get_chatbot_response(user_input):
    """
    Generate chatbot response with comprehensive error handling.
    
    Args:
        user_input: User's input message
        
    Returns:
        Enhanced chatbot response
    """
    global last_matched_intent
    
    try:
        # Input validation
        if not user_input or not isinstance(user_input, str):
            return "I didn't receive a valid message. Please try again! 🤔"
        
        if len(user_input.strip()) == 0:
            return "It looks like you sent an empty message. What would you like to know about fitness? 💪"
        
        if len(user_input) > 1000:  # Reasonable limit
            return "That's quite a long message! Could you please ask a shorter question? 😅"
        
        cleaned_input = clean_input(user_input)

        # Context handling (basic)
        if last_matched_intent == 'workout_plan':
            if re.search(r".*\b(beginner)\b.*", cleaned_input):
                last_matched_intent = 'beginner_workout'
                original_response = "Great! Start with squats, push-ups, and planks 💪."
            elif re.search(r".*\b(strength)\b.*", cleaned_input):
                last_matched_intent = 'strength_workout'
                original_response = "Strength training = squats, deadlifts, and presses. 🏋️"
            elif re.search(r".*\b(cardio)\b.*", cleaned_input):
                last_matched_intent = 'cardio_workout'
                original_response = "Cardio = running, cycling, swimming. ❤️"
            else:
                # Fall through to general rule matching
                original_response = None
        else:
            original_response = None

        # General rule matching if no context response
        if original_response is None:
            for pattern, rule_data in chatbot_rules.items():
                if pattern == "default":
                    continue
                try:
                    if re.search(pattern, cleaned_input):
                        last_matched_intent = rule_data['intent']
                        original_response = rule_data['response']
                        break
                except re.error as e:
                    logger.warning(f"Invalid regex pattern {pattern}: {str(e)}")
                    continue
            
            # Default response if no match
            if original_response is None:
                last_matched_intent = chatbot_rules["default"]['intent']
                original_response = chatbot_rules["default"]['response']

        # Get user ID from session state
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Enhance response with goal awareness (with error handling)
        try:
            enhanced_response = get_enhanced_chatbot_response(
                user_input, original_response, user_id, last_matched_intent
            )
            return enhanced_response
        except Exception as e:
            logger.error(f"Error enhancing chatbot response: {str(e)}")
            # Fallback to original response if enhancement fails
            return original_response
    
    except Exception as e:
        logger.error(f"Error in get_chatbot_response: {str(e)}")
        return "I encountered an error processing your message. Please try asking something else! 🤖"


# --- Goal Loading and Initialization ---
@handle_ui_errors("initialize user session")
def initialize_user_session():
    """Initialize user session with goal loading and data storage."""
    try:
        with safe_operation("initialize user session", 
                          loading_message="Setting up your fitness session..."):
            
            # Initialize data storage (this sets up user_id if not exists)
            data_storage = get_data_storage()
            
            # Get user ID from session state
            user_id = st.session_state.get('user_id', 'default_user')
            
            # Load user goal if not already loaded
            if 'user_goal_loaded' not in st.session_state:
                try:
                    goal_manager = get_goal_manager()
                    user_goal = goal_manager.get_user_goal(user_id)
                    st.session_state.user_goal = user_goal
                    st.session_state.user_goal_loaded = True
                    
                    # Check for goal reminders if user has a goal
                    if user_goal:
                        check_goal_reminders(user_goal)
                        
                except Exception as e:
                    logger.warning(f"Error loading user goal: {str(e)}")
                    st.session_state.user_goal = None
                    st.session_state.user_goal_loaded = True
                    # Don't show error to user as this is not critical for basic functionality
    
    except Exception as e:
        logger.error(f"Critical error initializing user session: {str(e)}")
        show_error("Failed to initialize your session", e, recoverable=True)

def check_goal_reminders(goal):
    """Check for goal deadline reminders and display them with enhanced feedback."""
    try:
        if not goal or not goal.is_active:
            return
        
        days_remaining = goal.days_remaining()
        
        # Show reminder for approaching deadlines
        if days_remaining <= 7 and days_remaining > 0:
            if 'deadline_reminder_shown' not in st.session_state:
                st.session_state.deadline_reminder_shown = True
                if days_remaining <= 3:
                    show_warning(f"Your {goal.goal_type.replace('_', ' ')} goal deadline is in {days_remaining} days!", 
                               "Consider reviewing your progress and making final adjustments.")
                else:
                    show_info(f"You have {days_remaining} days left to reach your {goal.goal_type.replace('_', ' ')} goal!", 
                             "You're in the final stretch - keep up the great work!")
        elif goal.is_overdue():
            if 'overdue_reminder_shown' not in st.session_state:
                st.session_state.overdue_reminder_shown = True
                show_warning(f"Your {goal.goal_type.replace('_', ' ')} goal deadline has passed", 
                           "Consider updating your timeline or setting a new goal to continue your progress.")
    
    except Exception as e:
        logger.error(f"Error checking goal reminders: {str(e)}")
        # Don't show error to user as this is not critical

def display_goal_milestone_celebrations():
    """Display any pending goal milestone celebrations with enhanced feedback."""
    try:
        user_id = st.session_state.get('user_id', 'default_user')
        user_goal = st.session_state.get('user_goal')
        
        if not user_goal or not user_goal.is_active:
            return
        
        # Calculate time-based milestones
        total_days = (user_goal.target_date - user_goal.created_date).days
        days_elapsed = total_days - user_goal.days_remaining()
        time_progress = (days_elapsed / total_days) if total_days > 0 else 0
        
        # Check for milestone celebrations (25%, 50%, 75% time elapsed)
        milestones = [0.25, 0.5, 0.75]
        milestone_key = f"milestone_{user_goal.goal_id}"
        
        celebrated_milestones = st.session_state.get(milestone_key, [])
        
        for milestone in milestones:
            if time_progress >= milestone and milestone not in celebrated_milestones:
                celebrated_milestones.append(milestone)
                st.session_state[milestone_key] = celebrated_milestones
                
                percentage = int(milestone * 100)
                
                # Enhanced milestone messages
                if percentage == 25:
                    show_success(f"🎉 Quarter milestone reached!", 
                               f"You're {percentage}% through your {user_goal.goal_type.replace('_', ' ')} timeline! Great start!")
                elif percentage == 50:
                    show_success(f"🎯 Halfway there!", 
                               f"You've completed {percentage}% of your {user_goal.goal_type.replace('_', ' ')} journey! Keep pushing!")
                elif percentage == 75:
                    show_success(f"🏆 Three-quarters complete!", 
                               f"You're {percentage}% through your {user_goal.goal_type.replace('_', ' ')} timeline! The finish line is in sight!")
                break
    
    except Exception as e:
        logger.error(f"Error displaying milestone celebrations: {str(e)}")
        # Don't show error to user as this is not critical

# --- Streamlit App ---
try:
    st.set_page_config(
        page_title="Fitness Assistant Bot", 
        page_icon="💪", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for consistent UI appearance and mobile responsiveness
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #495057;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Suggested prompts styling */
    .suggested-prompts {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .suggested-prompts h3 {
        color: #495057;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Form styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0px 12px;
            font-size: 0.9rem;
        }
        
        .suggested-prompts {
            padding: 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0px 8px;
            font-size: 0.8rem;
        }
    }
    
    /* Loading indicator styling */
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Chart container styling */
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main .block-container {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .css-1d391kg {
            background-color: #2d2d2d;
            border-right: 2px solid #404040;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2d2d2d;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            color: #ffffff;
        }
        
        .suggested-prompts {
            background-color: #2d2d2d;
            border: 1px solid #404040;
        }
        
        .metric-container {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            color: #ffffff;
        }
        
        .chart-container {
            background-color: #2d2d2d;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header with better styling
    st.markdown("""
    <div class="main-header">
        <h1>💪 Fitness Assistant Bot</h1>
        <p>Your personal AI companion for workouts, nutrition, and wellness</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize user session and load goals with error handling
    try:
        with loading_indicator("Initializing your fitness session...", "app_init"):
            initialize_user_session()
            initialize_goal_ui_session_state()
            display_goal_milestone_celebrations()
            
            # Mobile detection for responsive design
            st.markdown("""
            <script>
            function detectMobile() {
                return window.innerWidth <= 768;
            }
            
            if (detectMobile()) {
                window.parent.postMessage({type: 'mobile_detected'}, '*');
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Set mobile view flag (fallback method)
            if 'mobile_view' not in st.session_state:
                st.session_state.mobile_view = False
                
    except Exception as e:
        show_error("Failed to initialize the application", e, recoverable=True)
        st.stop()

    # Check and show onboarding flow for new users
    try:
        from fitness_goals.onboarding import check_and_show_onboarding
        onboarding_goal = check_and_show_onboarding()
        if onboarding_goal:
            # User just completed onboarding, refresh the page to show updated UI
            st.rerun()
    except Exception as e:
        logger.error(f"Error in onboarding flow: {str(e)}")
        # Don't show error to user as onboarding is not critical

    # Render goal setting sidebar with error boundary
    try:
        render_goal_setting_sidebar()
        
        # Add help section to sidebar
        with st.sidebar:
            st.markdown("---")
            with st.expander("ℹ️ Help & Tips"):
                st.markdown("""
                **🚀 Quick Start:**
                - Set a fitness goal using the form above
                - Ask me questions about workouts, nutrition, or sleep
                - Track your progress in the Progress tab
                
                **💡 Pro Tips:**
                - Use suggested prompts for quick answers
                - I provide personalized advice based on your goal
                - Check the Progress tab to log measurements
                
                **⌨️ Keyboard Shortcuts:**
                - `Enter` to send chat messages
                - `Ctrl+R` to refresh the app
                - `Esc` to clear current input
                
                **🔧 Troubleshooting:**
                - If something isn't working, try refreshing
                - Clear your data in the Data Management tab if needed
                - All data is stored locally in your browser
                """)
                
                if st.button("🔄 Refresh App", use_container_width=True):
                    st.rerun()
                    
    except Exception as e:
        logger.error(f"Error rendering goal sidebar: {str(e)}")
        with st.sidebar:
            show_error("Unable to load goal interface", e, recoverable=True)

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Progress Tracking", "🔧 Data Management"])

    with tab1:
        try:
            # --- Enhanced suggested prompts with better organization ---
            user_goal = st.session_state.get('user_goal')
            
            # Personalized prompts based on user's goal
            if user_goal and user_goal.is_active:
                goal_type = user_goal.goal_type.replace('_', ' ').title()
                personalized_prompts = {
                    "weight_loss": [
                        "Give me a fat-burning workout routine",
                        "What are the best foods for weight loss?",
                        "How can I stay motivated during weight loss?",
                        "Show me my weight loss progress"
                    ],
                    "muscle_building": [
                        "Create a muscle building workout plan",
                        "What should I eat to gain muscle?",
                        "How much protein do I need daily?",
                        "Track my muscle building progress"
                    ],
                    "weight_maintenance": [
                        "Give me a balanced workout routine",
                        "How do I maintain my current weight?",
                        "What's a good maintenance diet plan?",
                        "Show me my fitness progress"
                    ]
                }
                
                goal_specific_prompts = personalized_prompts.get(user_goal.goal_type, [])
                
                st.markdown(f"""
                <div class="suggested-prompts">
                    <h3>🎯 Personalized for Your {goal_type} Goal</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display goal-specific prompts
                cols = st.columns(2)
                for i, prompt in enumerate(goal_specific_prompts):
                    with cols[i % 2]:
                        if st.button(prompt, key=f"goal_prompt_{i}", use_container_width=True):
                            try:
                                st.session_state.messages.append({"role": "user", "content": prompt})
                                with loading_indicator(f"Personalizing response...", f"goal_response_{i}"):
                                    response = get_chatbot_response(prompt)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.rerun()
                            except Exception as e:
                                show_error("Failed to process personalized prompt", e, recoverable=True)
                
                st.markdown("---")
            
            # General suggested prompts
            general_prompts = [
                "Give me a beginner workout plan",
                "Suggest a healthy breakfast",
                "How much sleep do I need?",
                "Motivation tips to stay consistent",
                "What are good cardio exercises?",
                "Tell me about proteins and carbs",
                "Suggest a meal prep idea",
                "How many times should I workout per week?",
            ]

            st.markdown("""
            <div class="suggested-prompts">
                <h3>🔮 General Fitness Questions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Responsive grid for mobile
            if st.session_state.get('mobile_view', False):
                # Single column for mobile
                for i, prompt in enumerate(general_prompts):
                    if st.button(prompt, key=f"general_prompt_{i}", use_container_width=True):
                        try:
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            with loading_indicator(f"Thinking about: {prompt[:30]}...", f"response_{i}"):
                                response = get_chatbot_response(prompt)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.rerun()
                        except Exception as e:
                            show_error("Failed to process suggested prompt", e, recoverable=True)
            else:
                # Two columns for desktop
                cols = st.columns(2)
                for i, prompt in enumerate(general_prompts):
                    with cols[i % 2]:
                        if st.button(prompt, key=f"general_prompt_{i}", use_container_width=True):
                            try:
                                st.session_state.messages.append({"role": "user", "content": prompt})
                                with loading_indicator(f"Thinking about: {prompt[:30]}...", f"response_{i}"):
                                    response = get_chatbot_response(prompt)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.rerun()
                            except Exception as e:
                                show_error("Failed to process suggested prompt", e, recoverable=True)

            # Display chat messages with error handling
            try:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            except Exception as e:
                show_error("Error displaying chat history", e, recoverable=True)
                # Clear corrupted messages
                st.session_state.messages = []

            # Chat input with enhanced error handling
            if user_input := st.chat_input("Type your message..."):
                try:
                    # Validate input
                    if len(user_input.strip()) == 0:
                        show_warning("Please enter a message before sending")
                    elif len(user_input) > 1000:
                        show_warning("Message is too long. Please keep it under 1000 characters.")
                    else:
                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        with st.chat_message("user"):
                            st.markdown(user_input)

                        # Generate bot response with loading indicator
                        with loading_indicator("Generating response...", "chat_response"):
                            response = get_chatbot_response(user_input)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        
                        # Enhanced message history management
                        if len(st.session_state.messages) > 100:
                            # Keep first few messages (for context) and recent messages
                            first_messages = st.session_state.messages[:5]
                            recent_messages = st.session_state.messages[-45:]
                            st.session_state.messages = first_messages + recent_messages
                            show_info("💡 Chat history optimized for better performance", 
                                    "Keeping your recent conversations while maintaining app speed.")
                        
                        # Auto-scroll to bottom after new message
                        # Enhanced accessibility and keyboard shortcuts
                        st.markdown("""
                        <script>
                        // Auto-scroll to bottom after new message
                        setTimeout(function() {
                            var chatContainer = window.parent.document.querySelector('[data-testid="stChatMessageContainer"]');
                            if (chatContainer) {
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        }, 100);
                        
                        // Enhanced keyboard shortcuts
                        document.addEventListener('keydown', function(e) {
                            // Escape to clear input
                            if (e.key === 'Escape') {
                                var chatInput = window.parent.document.querySelector('[data-testid="stChatInput"] input');
                                if (chatInput) {
                                    chatInput.value = '';
                                    chatInput.focus();
                                }
                            }
                            
                            // Ctrl+/ for help
                            if (e.ctrlKey && e.key === '/') {
                                e.preventDefault();
                                var helpExpander = window.parent.document.querySelector('[data-testid="stExpander"] summary');
                                if (helpExpander && helpExpander.textContent.includes('Help')) {
                                    helpExpander.click();
                                }
                            }
                            
                            // Alt+1, Alt+2, Alt+3 for tab navigation
                            if (e.altKey && ['1', '2', '3'].includes(e.key)) {
                                e.preventDefault();
                                var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                                var tabIndex = parseInt(e.key) - 1;
                                if (tabs[tabIndex]) {
                                    tabs[tabIndex].click();
                                }
                            }
                        });
                        
                        // Focus management for accessibility
                        setTimeout(function() {
                            var chatInput = window.parent.document.querySelector('[data-testid="stChatInput"] input');
                            if (chatInput) {
                                chatInput.setAttribute('aria-label', 'Type your fitness question here');
                                chatInput.setAttribute('placeholder', 'Ask me about workouts, nutrition, or sleep...');
                            }
                        }, 500);
                        </script>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    show_error("Failed to process your message", e, recoverable=True)
                    logger.error(f"Error processing chat input: {str(e)}")
        
        except Exception as e:
            show_error("Error loading chat interface", e, recoverable=True)

    with tab2:
        # Progress tracking interface with error boundary
        try:
            render_progress_tracking_interface()
        except Exception as e:
            show_error("Failed to load progress tracking interface", e, recoverable=True)
            logger.error(f"Error in progress tracking tab: {str(e)}")

    with tab3:
        # Data management interface with error boundary
        try:
            from fitness_goals.data_management_ui import render_data_management_ui
            render_data_management_ui()
        except Exception as e:
            show_error("Failed to load data management interface", e, recoverable=True)
            logger.error(f"Error in data management tab: {str(e)}")

except Exception as e:
    # Critical application error
    st.error("🚨 **Critical Application Error**")
    st.error(f"The application encountered a critical error: {str(e)}")
    st.info("Please refresh the page to restart the application.")
    logger.critical(f"Critical application error: {str(e)}")
    
    # Show recovery options
    if st.button("🔄 Restart Application"):
        st.rerun()
    
    if st.button("🗑️ Clear All Data and Restart"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Enhanced footer with app information
st.markdown("---")
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #6c757d;
    font-size: 0.9rem;
    border-top: 1px solid #e9ecef;
    margin-top: 2rem;
">
    <p style="margin: 0.5rem 0;">
        <strong>💪 Fitness Assistant Bot</strong> v2.0 | 
        Built with Streamlit & Python | 
        <span style="color: #28a745;">●</span> All data stored locally
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.8rem;">
        🎯 Goal Setting • 📊 Progress Tracking • 🤖 AI-Powered Recommendations • 📱 Mobile Friendly
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.8rem;">
        Need help? Use <kbd>Ctrl+/</kbd> for shortcuts or check the sidebar help section
    </p>
</div>
""", unsafe_allow_html=True)
