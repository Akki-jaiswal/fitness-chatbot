#!/usr/bin/env python3
"""
Demo script showcasing all 14 enhanced features of the fitness chatbot
Run this to see all implemented features in action
"""

from enhanced_bot import EnhancedFitnessBot, UserProfile
import datetime

def demo_all_features():
    print("🚀 ENHANCED FITNESS CHATBOT - FEATURE DEMO")
    print("=" * 50)
    
    bot = EnhancedFitnessBot()
    profile = UserProfile()
    
    # Feature 1: Dynamic Workout Difficulty Adjustment
    print("\n1️⃣ DYNAMIC WORKOUT DIFFICULTY ADJUSTMENT")
    print("-" * 40)
    
    test_inputs = [
        "I'm tired, give me a quick workout",
        "I'm energetic and have 45 minutes",
        "I'm feeling okay, 20 minute workout"
    ]
    
    for input_text in test_inputs:
        context = bot.extract_context(input_text)
        response = bot.generate_dynamic_workout(context, profile)
        print(f"Input: '{input_text}'")
        print(f"Response: {response}\n")
    
    # Feature 2: Contextual Time-based Recommendations
    print("2️⃣ CONTEXTUAL TIME-BASED RECOMMENDATIONS")
    print("-" * 40)
    
    time_inputs = [
        "I have 5 minutes",
        "Quick 15 minute workout",
        "I have an hour to exercise"
    ]
    
    for input_text in time_inputs:
        context = bot.extract_context(input_text)
        response = bot.generate_dynamic_workout(context, profile)
        print(f"Input: '{input_text}'")
        print(f"Response: {response}\n")
    
    # Feature 3: Meal Planning with Constraints
    print("3️⃣ MEAL PLANNING WITH CONSTRAINTS")
    print("-" * 40)
    
    meal_inputs = [
        "I need a budget breakfast",
        "Suggest an expensive dinner",
        "What's a good lunch?"
    ]
    
    for input_text in meal_inputs:
        context = bot.extract_context(input_text)
        response = bot.generate_meal_suggestion(context, profile)
        print(f"Input: '{input_text}'")
        print(f"Response: {response}\n")
    
    # Feature 4: Hydration Intelligence
    print("4️⃣ HYDRATION INTELLIGENCE")
    print("-" * 40)
    
    hydration_contexts = [
        {"energy": 8},  # High energy (workout day)
        {"energy": 4},  # Low energy (rest day)
        {}  # Normal day
    ]
    
    for context in hydration_contexts:
        response = bot.hydration_intelligence(context)
        print(f"Context: {context}")
        print(f"Response: {response}\n")
    
    # Feature 5: Mood-Fitness Correlation Tracker
    print("5️⃣ MOOD-FITNESS CORRELATION TRACKER")
    print("-" * 40)
    
    moods = ["stressed", "sad", "anxious", "angry", "tired", "excited"]
    
    for mood in moods:
        response = bot.mood_fitness_correlation(mood, profile)
        print(f"Mood: {mood}")
        print(f"Response: {response}\n")
    
    # Feature 6: Smart Goal Setting System
    print("6️⃣ SMART GOAL SETTING SYSTEM")
    print("-" * 40)
    
    vague_goals = [
        "I want to lose weight",
        "I want to get fit",
        "I want to build muscle",
        "I want to eat healthy"
    ]
    
    for goal in vague_goals:
        response = bot.generate_smart_goals(goal)
        print(f"Vague Goal: '{goal}'")
        print(f"SMART Goal: {response}\n")
    
    # Feature 7: Conversational Memory System
    print("7️⃣ CONVERSATIONAL MEMORY SYSTEM")
    print("-" * 40)
    
    # Simulate conversation memory
    bot.conversation_memory = [
        {
            'timestamp': datetime.datetime.now(),
            'user': 'Give me a workout',
            'bot': 'Here is a workout plan...',
            'context': {'energy': 7}
        }
    ]
    
    print("Previous conversation stored:")
    print(f"User said: {bot.conversation_memory[0]['user']}")
    print(f"Bot responded: {bot.conversation_memory[0]['bot']}")
    print(f"Context remembered: {bot.conversation_memory[0]['context']}\n")
    
    # Feature 8: Multi-modal Input Processing
    print("8️⃣ MULTI-MODAL INPUT PROCESSING")
    print("-" * 40)
    
    emoji_inputs = [
        "I'm feeling 😴 today",
        "Ready to workout 💪",
        "Need some motivation 😔"
    ]
    
    for input_text in emoji_inputs:
        context = bot.extract_context(input_text)
        print(f"Input: '{input_text}'")
        print(f"Extracted context: {context}\n")
    
    # Feature 9: Adaptive Circuit Builder
    print("9️⃣ ADAPTIVE CIRCUIT BUILDER")
    print("-" * 40)
    
    circuit_contexts = [
        {"time": 10, "energy": 8},  # Quick high-intensity
        {"time": 30, "energy": 5},  # Medium workout
        {"time": 60, "energy": 7}   # Long workout
    ]
    
    for context in circuit_contexts:
        response = bot.generate_dynamic_workout(context, profile)
        print(f"Context: {context}")
        print(f"Circuit: {response}\n")
    
    # Feature 10: Exercise Variation Engine
    print("🔟 EXERCISE VARIATION ENGINE")
    print("-" * 40)
    
    exercises = ["squats", "pushups", "planks"]
    
    for exercise in exercises:
        # First time doing exercise
        response1 = bot.exercise_variation_engine(exercise, profile)
        print(f"First time doing {exercise}: {response1}")
        
        # Second time (should suggest variation)
        response2 = bot.exercise_variation_engine(exercise, profile)
        print(f"Repeat {exercise}: {response2}\n")
    
    # Feature 11: Challenge System
    print("1️⃣1️⃣ CHALLENGE SYSTEM")
    print("-" * 40)
    
    for i in range(3):
        challenge = bot.generate_weekly_challenge()
        print(f"Challenge {i+1}: {challenge}\n")
    
    # Feature 12: Social Accountability (Simulated)
    print("1️⃣2️⃣ STREAK TRACKING & ACCOUNTABILITY")
    print("-" * 40)
    
    # Simulate workout streak
    for day in range(1, 8):
        streak_msg = bot.track_streaks("workout", profile)
        print(f"Day {day}: {streak_msg}")
    
    print()
    
    # Feature 13: Intelligent Reminder System
    print("1️⃣3️⃣ INTELLIGENT REMINDER SYSTEM")
    print("-" * 40)
    
    reminder = bot.intelligent_reminders(profile)
    print(f"Current reminder: {reminder}\n")
    
    # Feature 14: Recovery & Rest Optimization
    print("1️⃣4️⃣ RECOVERY & REST OPTIMIZATION")
    print("-" * 40)
    
    recovery_contexts = [
        {"energy": 2},  # Very tired
        {"energy": 5},  # Normal
        {"energy": 8}   # High energy
    ]
    
    # Simulate long workout streak for active recovery
    profile.workout_streak = 6
    
    for context in recovery_contexts:
        response = bot.recovery_optimization(context, profile)
        print(f"Energy level {context['energy']}/10: {response}\n")
    
    print("🎉 ALL 14 ENHANCED FEATURES DEMONSTRATED!")
    print("=" * 50)
    print("✅ Dynamic Workout Difficulty Adjustment")
    print("✅ Contextual Time-based Recommendations") 
    print("✅ Meal Planning with Constraints")
    print("✅ Hydration Intelligence")
    print("✅ Mood-Fitness Correlation Tracker")
    print("✅ Smart Goal Setting System")
    print("✅ Conversational Memory System")
    print("✅ Multi-modal Input Processing")
    print("✅ Adaptive Circuit Builder")
    print("✅ Exercise Variation Engine")
    print("✅ Challenge System")
    print("✅ Streak Tracking & Social Accountability")
    print("✅ Intelligent Reminder System")
    print("✅ Recovery & Rest Optimization")

if __name__ == "__main__":
    demo_all_features()