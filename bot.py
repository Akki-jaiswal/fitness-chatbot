import re
from rapidfuzz import fuzz, process
# --- Chatbot Rules Definition for a Fitness App ---
# Maps user input patterns (keywords or regex) to chatbot responses and their intent.
# Rules are processed in order, so specific patterns should appear before general ones.
chatbot_rules = {
    # Greetings and Introductions
    r".*\b(hi|hello|hey|greetings)\b.*": {'response': "Hello there! Welcome to your fitness companion. How can I help you achieve your health goals today? You can ask me about workouts, meals, or sleep.", 'intent': 'greeting'},
    r".*\b(how are you|how's it going)\b.*": {'response': "I'm a bot designed to help you with your fitness journey! How are you feeling today? Feel free to ask about workouts, nutrition, or sleep.", 'intent': 'greeting'},
    r".*\b(what is your name|who are you)\b.*": {'response': "I'm your friendly Fitness Bot, here to answer your questions about health, workouts, and nutrition. What would you like to know?", 'intent': 'greeting'},

    # Workout-related Questions
    # Regex for flexible matching of workout plan queries
    r".*\b(workout plan|exercise routine|gym plan|workout|exercise|gym)\b.*": {'response': "I can help with workout plans! Are you looking for something for beginners, strength training, cardio, or flexibility? You can specify your preference.", 'intent': 'workout_plan'},
    # Regex for beginner-related workout queries
    r".*\b(beginner|start exercising)\b.*": {'response': "For beginners, I recommend starting with bodyweight exercises like squats, push-ups (on knees if needed), and planks. Consistency is key! You can also ask about strength training or cardio.", 'intent': 'beginner_workout'},
    # Regex for strength training queries
    r".*\b(strength|strength training|build muscle)\b.*": {'response': "Strength training is great! Focus on compound movements like squats, deadlifts, bench press, and overhead press. Ensure proper form! You can also ask about workout frequency or warm-up routines.", 'intent': 'strength_workout'},
    # Regex for cardio/endurance queries
    r".*\b(cardio|endurance)\b.*": {'response': "Cardio helps heart health! Consider running, cycling, swimming, or brisk walking. Aim for at least 30 minutes most days. You can also ask about warm-up or cool-down exercises.", 'intent': 'cardio_workout'},
    # Warm-up and cool-down queries
    r".*\b(warm up|cool down|warmup|cooldown)\b.*": {'response': "Always warm up for 5-10 minutes before and cool down for 5-10 minutes after your workout to prevent injuries. You can also ask about different workout types.", 'intent': 'workout_prep'},
    # Workout frequency queries
    r".*\b(how many times a week|workout frequency|frequency)\b.*": {'response': "For general fitness, aim for 3-5 workout sessions per week, allowing for rest days. You can also ask about specific exercises.", 'intent': 'workout_frequency'},

    # Meal and Nutrition Questions
    # Regex for general meal/nutrition queries
    r".*\b(healthy meals|diet plan|nutrition advice|meals|diet|nutrition)\b.*": {'response': "Nutrition is vital! Are you interested in meal ideas for breakfast, lunch, dinner, or snacks? You can specify your meal preference.", 'intent': 'nutrition_plan'},
    # Regex to match breakfast-related queries
    r".*\b(healthy breakfast|breakfast ideas|breakfast)\b.*": {'response': "A healthy breakfast could be oatmeal with fruits and nuts, Greek yogurt with berries, or scrambled eggs with veggies. You can also ask about meal prepping or other meal ideas.", 'intent': 'breakfast_ideas'},
    # Regex for meal prep queries
    r".*\b(meal prep|prepare food|mealprep)\b.*": {'response': "Meal prepping can save time! Choose a day to cook larger batches of healthy proteins, complex carbs and veggies for the week. You can also ask about specific macronutrients.", 'intent': 'meal_prep'},
    # Rule for weight/calorie queries (generic response only, as personalized advice needs more advanced systems)
    r".*\b(calorie intake|how many calories|kg|kilogram|kgs|weight)\b.*": {'response': "Calorie needs and weight management are very personal. It's best to consult a professional for tailored advice. I can tell you about general nutrition principles like protein, carbs, and fats.", 'intent': 'weight_calories'},
    # Macronutrient queries
    r".*\b(protein|carbs|fats)\b.*": {'response': "Balanced nutrition includes protein for muscle repair, complex carbs for energy, and healthy fats for overall health. Focus on whole, unprocessed foods. You can also ask about healthy meal ideas.", 'intent': 'macros'},

    # Sleep-related Questions
    # Sleep improvement tips
    r".*\b(improve sleep|sleep better|sleep tips|sleep)\b.*": {'response': "Quality sleep is crucial for recovery! Try to maintain a consistent sleep schedule, create a dark and quiet environment, and avoid screens before bed. You can also ask about recommended sleep duration.", 'intent': 'sleep_tips'},
    # Sleep duration queries
    r".*\b(how much sleep|hours of sleep)\b.*": {'response': "Most adults need 7-9 hours of quality sleep per night for optimal health and recovery. You can also ask for tips to improve your sleep.", 'intent': 'sleep_duration'},
    # Insomnia help
    r".*\b(insomnia|can't sleep)\b.*": {'response': "If you're struggling with insomnia, try relaxation techniques, limit caffeine, and ensure your bedroom is conducive to sleep. If it persists, consider seeing a doctor. You can also ask about general sleep tips.", 'intent': 'insomnia_help'},

    # General Health & App Features
    r".*\b(track progress|monitor goals|track|progress|goals)\b.*": {'response': "Our app allows you to track your workouts, log your meals, and monitor your sleep patterns to see your progress over time! You can ask me about other app features.", 'intent': 'app_features'},
    # Regex for app features queries
    r".*\b(app features|what can this app do|features)\b.*": {'response': "This app helps you with personalized workout plans, meal tracking, sleep monitoring, and goal setting to support your fitness journey. You can ask about specific features like tracking.", 'intent': 'app_features'},
    # Motivation queries
    r".*\b(motivation|stay motivated)\b.*": {'response': "Staying motivated can be tough! Set small, achievable goals, find an accountability partner, and celebrate your progress. You can also ask about tracking your progress.", 'intent': 'motivation'},

    # Polite Closings and Exit Phrases
    r".*\b(thank you|thanks)\b.*": {'response': "You're most welcome! Keep up the great work on your fitness journey! Feel free to ask anything else.", 'intent': 'thank_you'},
    r".*\b(bye|goodbye|exit|quit|see you)\b.*": {'response': "Goodbye! Stay fit, stay healthy, and remember consistency is key! You can always come back if you have more fitness questions.", 'intent': 'exit'},

    # Default Response (when no specific rule matches)
    "default": {'response': "I'm not sure I understand that fitness-related question. You can ask me about workouts, meals, or sleep.", 'intent': 'unknown'}
}

# Global variable to store the last matched intent for simple context tracking.
last_matched_intent = None

def clean_input(user_input):
    """
    Cleans user input by converting to lowercase and removing punctuation for reliable matching.
    """
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input) # Remove non-alphanumeric/whitespace chars
    return user_input

def get_chatbot_response(user_input):
    """
    Determines the chatbot's response based on predefined rules and simple context.
    Simulates basic NLP by matching patterns and considering the previous turn.
    """
    global last_matched_intent

    cleaned_input = clean_input(user_input)

    # --- Context-specific responses ---
    # Prioritizes responses based on the last conversation turn.
    # Note: These context-specific responses are hardcoded for simplicity.
    # For more complex context, the 'response' from chatbot_rules could be used.
     # --- Context-specific responses ---
    if last_matched_intent == 'workout_plan':
        context_keywords = {
            "beginner": ("beginner_workout",
                         "Great! For beginners, focus on bodyweight exercises like squats, push-ups, and planks. Start slow and build consistency. You can also ask about strength training or cardio workouts."),
            "strength": ("strength_workout",
                         "Excellent! For strength training, compound movements like squats, deadlifts, and bench presses are key. Remember proper form! You can also ask about workout frequency or warm-up routines."),
            "cardio": ("cardio_workout",
                       "Good choice! Cardio can include running, cycling, or swimming. Aim for a consistent duration. You can also ask about warm-up or cool-down exercises.")
        }

        # 1) Try exact regex match first
        for kw, (intent, resp) in context_keywords.items():
            if re.search(rf".*\b({kw})\b.*", cleaned_input):
                last_matched_intent = intent
                return resp

        # 2) 🔹 If no regex match, try fuzzy match
        match, score, _ = process.extractOne(cleaned_input, context_keywords.keys(), scorer=fuzz.ratio)
        if match and score >= 80:
            intent, resp = context_keywords[match]
            last_matched_intent = intent
            return f"(Did you mean '{match}'?) " + resp
    # Add more context checks here if needed for other intents

    # --- General rule matching (fallback if no context match) ---
    for pattern, rule_data in chatbot_rules.items():
        # Check for regex patterns first, as they are more flexible
        if isinstance(pattern, str) and pattern.startswith(".*") and pattern.endswith(".*"):
            if re.search(pattern, cleaned_input):
                last_matched_intent = rule_data['intent']
                return rule_data['response']
        # Fallback to direct keyword tuples if no regex match (though most are now regex)
        elif isinstance(pattern, tuple):
            for keyword in pattern:
                if keyword in cleaned_input:
                    last_matched_intent = rule_data['intent']
                    return rule_data['response']
                
         # --- 🔹 Fuzzy Matching Fallback ---
    keywords = []
    for pattern in chatbot_rules.keys():
        if pattern != "default":
            keywords.extend(re.findall(r"\b\w+\b", pattern))  # extract words from regex

    # Compare each word in input to keywords
    for word in cleaned_input.split():
        match, score, _ = process.extractOne(word, keywords, scorer=fuzz.ratio)
        if match and score >= 80:  # threshold
            for pattern, rule_data in chatbot_rules.items():
                if re.search(match, pattern):
                    last_matched_intent = rule_data['intent']
                    return f"(Did you mean '{match}'?) " + rule_data['response']
    
    

    # Default response if no rule or context matches
    last_matched_intent = chatbot_rules["default"]['intent']
    return chatbot_rules["default"]['response']

def run_chatbot():
    """
    Starts the interactive chatbot session in the console.
    This is the main loop where the user types input and gets responses.
    """
    print("--- Fitness App Chatbot ---")
    print("Ask me about workouts, meals, or sleep! Type 'bye' to exit.")
    print("-------------------------------------------------")

    while True:
        user_message = input("You: ")
        # Check for exit commands
        if user_message.lower() in ("bye", "goodbye", "exit", "quit", "see you"):
            # The exit response already contains the instruction, so we just print it.
            print(f"Chatbot: {get_chatbot_response(user_message)}")
            break
        else:
            # Get and print the chatbot's response
            response = get_chatbot_response(user_message)
            print(f"Chatbot: {response}")

# Ensures `run_chatbot()` is called only when the script is executed directly.
if __name__ == "__main__":
    run_chatbot()
