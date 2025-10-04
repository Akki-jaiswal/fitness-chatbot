import re
import streamlit as st
from enhanced_bot import get_enhanced_response, get_fallback_response

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


def get_chatbot_response(user_input):
    try:
        # Try enhanced response first
        enhanced_response = get_enhanced_response(user_input)
        if enhanced_response and "I'm here to help" not in enhanced_response:
            return enhanced_response
    except Exception as e:
        st.error(f"Enhanced features temporarily unavailable: {str(e)}")
    
    # Fallback to original logic
    global last_matched_intent
    cleaned_input = clean_input(user_input)

    # Context handling (basic)
    if last_matched_intent == 'workout_plan':
        if re.search(r".*\b(beginner)\b.*", cleaned_input):
            last_matched_intent = 'beginner_workout'
            return "Great! Start with squats, push-ups, and planks 💪."
        elif re.search(r".*\b(strength)\b.*", cleaned_input):
            last_matched_intent = 'strength_workout'
            return "Strength training = squats, deadlifts, and presses. 🏋️"
        elif re.search(r".*\b(cardio)\b.*", cleaned_input):
            last_matched_intent = 'cardio_workout'
            return "Cardio = running, cycling, swimming. ❤️"

    # General rule matching
    for pattern, rule_data in chatbot_rules.items():
        if pattern == "default":
            continue
        if re.search(pattern, cleaned_input):
            last_matched_intent = rule_data['intent']
            return rule_data['response']

    # Default response
    last_matched_intent = chatbot_rules["default"]['intent']
    return chatbot_rules["default"]['response']


# --- Streamlit App ---
st.set_page_config(page_title="Fitness Chatbot", page_icon="💪")
st.title("💬 Enhanced Fitness Assistant Bot")
st.write("🚀 **New Features**: Dynamic workouts, smart meal planning, streak tracking, mood-based fitness, challenges & more!")

# Display user profile info
if 'user_profile' in st.session_state:
    profile = st.session_state.user_profile
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Workout Streak", f"{profile.workout_streak} days")
    with col2:
        st.metric("Energy Level", f"{profile.energy_level}/10")
    with col3:
        st.metric("Available Time", f"{profile.available_time} min")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Enhanced suggested prompts ---
suggested_prompts = [
    "I'm tired, give me a quick 10-minute workout",
    "I'm energetic and have 30 minutes to exercise",
    "Suggest a budget-friendly healthy breakfast",
    "I need a weekly fitness challenge",
    "Help me set a SMART fitness goal",
    "I'm stressed, what exercise should I do?",
    "Track my workout streak",
    "Give me hydration tips for today",
    "I need recovery suggestions",
    "Suggest seasonal meal ideas"
]

st.markdown("### 🔮 Try These Enhanced Features")
cols = st.columns(2)
for i, prompt in enumerate(suggested_prompts):
    if cols[i % 2].button(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_chatbot_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate bot response
    response = get_chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
