import re
import streamlit as st
import random
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class UserProfile:
    energy_level: int = 5
    available_time: int = 30
    equipment: List[str] = field(default_factory=lambda: ["bodyweight"])
    fitness_level: str = "beginner"
    last_workout_date: Optional[str] = None
    workout_streak: int = 0
    nutrition_streak: int = 0
    preferred_exercises: List[str] = field(default_factory=list)
    budget_range: str = "medium"
    goals: List[str] = field(default_factory=list)
    mood_history: List[Dict] = field(default_factory=list)

class EnhancedFitnessBot:
    def __init__(self):
        self.conversation_memory = []
        self.workout_variations = {
            "squats": ["jump squats", "sumo squats", "single-leg squats", "wall squats"],
            "pushups": ["incline pushups", "diamond pushups", "wide-grip pushups", "knee pushups"],
            "planks": ["side planks", "plank up-downs", "mountain climber planks", "reverse planks"]
        }
        self.seasonal_foods = {
            "winter": ["soup", "stew", "roasted vegetables", "warm oatmeal"],
            "spring": ["fresh salads", "asparagus", "strawberries", "light soups"],
            "summer": ["cold gazpacho", "grilled vegetables", "fresh fruits", "smoothie bowls"],
            "fall": ["pumpkin dishes", "apple recipes", "hearty grains", "warm spices"]
        }
        
    def extract_context(self, user_input: str) -> Dict:
        """Extract context from user input"""
        context = {}
        
        # Energy level detection
        if any(word in user_input.lower() for word in ["tired", "exhausted", "drained", "😴"]):
            context["energy"] = 3
        elif any(word in user_input.lower() for word in ["energetic", "pumped", "motivated", "💪"]):
            context["energy"] = 8
        elif any(word in user_input.lower() for word in ["okay", "normal", "fine"]):
            context["energy"] = 5
            
        # Time extraction
        time_patterns = [r"(\d+)\s*min", r"quick", r"short", r"long", r"hour"]
        for pattern in time_patterns:
            if re.search(pattern, user_input.lower()):
                if "quick" in user_input.lower() or "short" in user_input.lower():
                    context["time"] = 10
                elif "long" in user_input.lower() or "hour" in user_input.lower():
                    context["time"] = 60
                else:
                    match = re.search(r"(\d+)", user_input)
                    if match:
                        context["time"] = int(match.group(1))
        
        # Budget detection
        if any(word in user_input.lower() for word in ["budget", "cheap", "affordable", "money"]):
            context["budget"] = "low"
        elif any(word in user_input.lower() for word in ["expensive", "premium", "high-end"]):
            context["budget"] = "high"
            
        # Equipment detection
        equipment_keywords = ["dumbbells", "resistance bands", "yoga mat", "no equipment", "bodyweight"]
        for eq in equipment_keywords:
            if eq in user_input.lower():
                context["equipment"] = eq
                
        return context

    def generate_dynamic_workout(self, context: Dict, profile: 'UserProfile') -> str:
        """Generate adaptive workout based on context and profile"""
        energy = context.get("energy", profile.energy_level)
        time = context.get("time", profile.available_time)
        equipment = context.get("equipment", "bodyweight")
        
        # Adjust difficulty based on energy
        if energy <= 4:
            exercises = ["gentle stretching", "light yoga", "walking", "easy bodyweight movements"]
            intensity = "Low"
        elif energy >= 7:
            exercises = ["HIIT circuit", "burpees", "jump squats", "mountain climbers"]
            intensity = "High"
        else:
            exercises = ["squats", "push-ups", "lunges", "planks"]
            intensity = "Moderate"
            
        # Time-based modifications
        if time <= 10:
            workout_type = "Quick Burst"
            reps = "30 seconds each"
        elif time <= 20:
            workout_type = "Express"
            reps = "45 seconds each, 15s rest"
        else:
            workout_type = "Full"
            reps = "3 sets of 12-15 reps"
            
        selected_exercise = random.choice(exercises)
        
        # Add variation if user has done this before
        if selected_exercise in profile.preferred_exercises:
            if selected_exercise in self.workout_variations:
                variation = random.choice(self.workout_variations[selected_exercise])
                selected_exercise = f"{variation} (variation of {selected_exercise})"
        
        return f"🏋️ {workout_type} {intensity}-Intensity Workout ({time} min):\n{selected_exercise} - {reps}\n💡 Energy level: {energy}/10"

    def generate_meal_suggestion(self, context: Dict, profile: 'UserProfile') -> str:
        """Generate contextual meal suggestions"""
        current_hour = datetime.datetime.now().hour
        budget = context.get("budget", profile.budget_range)
        
        # Determine meal type
        if current_hour < 10:
            meal_type = "breakfast"
        elif current_hour < 15:
            meal_type = "lunch"
        else:
            meal_type = "dinner"
            
        # Budget-based suggestions
        budget_meals = {
            "low": {
                "breakfast": "Oatmeal with banana and peanut butter (~$1.50)",
                "lunch": "Lentil soup with whole grain bread (~$2.00)",
                "dinner": "Rice and beans with vegetables (~$2.50)"
            },
            "medium": {
                "breakfast": "Greek yogurt with berries and granola (~$3.00)",
                "lunch": "Quinoa salad with chickpeas (~$4.00)",
                "dinner": "Grilled chicken with sweet potato (~$5.00)"
            },
            "high": {
                "breakfast": "Avocado toast with smoked salmon (~$8.00)",
                "lunch": "Organic salad with grass-fed beef (~$12.00)",
                "dinner": "Wild-caught fish with quinoa (~$15.00)"
            }
        }
        
        # Seasonal suggestions
        season = self.get_current_season()
        seasonal_ingredient = random.choice(self.seasonal_foods[season])
        
        base_meal = budget_meals[budget][meal_type]
        
        return f"🍽️ {meal_type.title()} Suggestion:\n{base_meal}\n🌿 Seasonal twist: Add {seasonal_ingredient}\n💰 Budget: {budget.title()}"

    def get_current_season(self) -> str:
        """Determine current season"""
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def track_streaks(self, activity_type: str, profile: 'UserProfile') -> str:
        """Track and celebrate streaks"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        if activity_type == "workout":
            if profile.last_workout_date:
                last_date = datetime.datetime.strptime(profile.last_workout_date, "%Y-%m-%d")
                if (datetime.datetime.now() - last_date).days == 1:
                    profile.workout_streak += 1
                else:
                    profile.workout_streak = 1
            else:
                profile.workout_streak = 1
            profile.last_workout_date = today
            streak = profile.workout_streak
        else:  # nutrition
            profile.nutrition_streak += 1
            streak = profile.nutrition_streak
            
        # Celebration messages
        if streak == 1:
            return f"🎯 Day 1 of your {activity_type} journey!"
        elif streak == 3:
            return f"🔥 3-day streak! You're building momentum!"
        elif streak == 7:
            return f"⭐ 1 week streak! Habit forming!"
        elif streak == 30:
            return f"🏆 30-day streak! You're unstoppable!"
        else:
            return f"💪 Day {streak} streak! Keep going!"

    def generate_smart_goals(self, user_input: str) -> str:
        """Convert vague goals to SMART goals"""
        goal_templates = {
            "lose weight": "Lose 1-2 pounds per week through 150 minutes of cardio + strength training 3x/week",
            "get fit": "Complete 30-minute workouts 4 times per week for the next 8 weeks",
            "build muscle": "Increase strength by 10% in major lifts over 12 weeks with progressive overload",
            "eat healthy": "Eat 5 servings of fruits/vegetables daily and meal prep 3 days per week",
            "sleep better": "Maintain 7-8 hours sleep nightly with consistent bedtime for 4 weeks"
        }
        
        for vague_goal, smart_goal in goal_templates.items():
            if vague_goal in user_input.lower():
                return f"🎯 SMART Goal Conversion:\nFrom: '{vague_goal}'\nTo: '{smart_goal}'\n📅 Let's break this into weekly milestones!"
                
        return "🎯 Let's make your goal SMART (Specific, Measurable, Achievable, Relevant, Time-bound)! What exactly do you want to achieve?"

    def mood_fitness_correlation(self, mood: str, profile: 'UserProfile') -> str:
        """Track mood and suggest appropriate activities"""
        mood_activities = {
            "stressed": "Try gentle yoga or a 10-minute walk to reduce cortisol levels",
            "sad": "Light cardio like dancing can boost endorphins naturally",
            "anxious": "Deep breathing exercises combined with stretching",
            "angry": "High-intensity workout to channel energy positively",
            "tired": "Gentle movement like tai chi or light stretching",
            "excited": "Perfect energy for a challenging HIIT workout!"
        }
        
        # Log mood
        profile.mood_history.append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "mood": mood,
            "activity_suggested": mood_activities.get(mood, "balanced workout")
        })
        
        return f"😊 Mood-based suggestion: {mood_activities.get(mood, 'balanced workout')}"

    def hydration_intelligence(self, context: Dict) -> str:
        """Smart hydration recommendations"""
        base_water = 8  # glasses
        current_hour = datetime.datetime.now().hour
        
        # Time-based adjustments
        if current_hour < 12:
            timing = "Start your day with 2 glasses of water"
        elif current_hour < 18:
            timing = "Afternoon hydration: 1 glass every hour"
        else:
            timing = "Evening: Light hydration, avoid overdrinking before bed"
            
        # Activity adjustments
        if context.get("energy", 5) >= 7:  # High energy = likely workout
            base_water += 2
            activity_note = "Extra 2 glasses for workout recovery"
        else:
            activity_note = "Standard daily intake"
            
        return f"💧 Hydration Plan: {base_water} glasses today\n⏰ {timing}\n🏃 {activity_note}"

    def exercise_variation_engine(self, exercise: str, profile: 'UserProfile') -> str:
        """Prevent boredom with exercise variations"""
        if exercise in profile.preferred_exercises:
            if exercise in self.workout_variations:
                variations = self.workout_variations[exercise]
                new_variation = random.choice(variations)
                return f"🔄 Variation Alert! Instead of regular {exercise}, try: {new_variation}"
        
        # Add to preferred exercises
        if exercise not in profile.preferred_exercises:
            profile.preferred_exercises.append(exercise)
            
        return f"✅ Added {exercise} to your exercise history for future variations!"

    def generate_weekly_challenge(self) -> str:
        """Create engaging weekly challenges"""
        challenges = [
            "Stair Master: Take stairs instead of elevators all week",
            "Phone Fitness: 10 squats every time you check your phone",
            "Hydration Hero: Drink water before every meal",
            "Plank Power: Hold a 2-minute plank 3 times this week",
            "Lunch Walker: 10-minute walk after lunch daily",
            "Morning Mover: 5-minute stretch routine every morning",
            "Snack Swapper: Replace one unhealthy snack daily with fruit"
        ]
        
        challenge = random.choice(challenges)
        return f"🎯 This Week's Challenge: {challenge}\n🏆 Complete it for bonus motivation points!"

    def intelligent_reminders(self, profile: 'UserProfile') -> str:
        """Context-aware reminders"""
        current_hour = datetime.datetime.now().hour
        day_of_week = datetime.datetime.now().strftime("%A")
        
        if current_hour == 7:
            return "🌅 Morning Reminder: Perfect time for a energizing workout!"
        elif current_hour == 12:
            return "🥗 Lunch Reminder: How about a healthy meal and a short walk?"
        elif current_hour == 18:
            return "🌆 Evening Reminder: Wind down with some gentle stretching"
        elif day_of_week == "Monday":
            return "💪 Monday Motivation: Start the week strong with your fitness goals!"
        elif day_of_week == "Friday":
            return "🎉 Friday Focus: Finish the week with a rewarding workout!"
        
        return "⏰ Stay consistent with your fitness routine today!"

    def recovery_optimization(self, context: Dict, profile: 'UserProfile') -> str:
        """Smart recovery suggestions"""
        if context.get("energy", 5) <= 3:
            return "🛌 Recovery Day Suggestions:\n• Gentle stretching (10 min)\n• Foam rolling\n• Light walk\n• Focus on sleep quality"
        elif profile.workout_streak >= 5:
            return "⚡ Active Recovery:\n• Yoga flow\n• Swimming\n• Light bike ride\n• Meditation + stretching"
        else:
            return "🔄 Recovery Integration:\n• 5-min cool down after workouts\n• Stretch major muscle groups\n• Stay hydrated"

# Enhanced Streamlit Integration
def get_enhanced_response(user_input: str) -> str:
    """Main function to get enhanced chatbot response"""
    if 'enhanced_bot' not in st.session_state:
        st.session_state.enhanced_bot = EnhancedFitnessBot()
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = UserProfile()
    
    bot = st.session_state.enhanced_bot
    profile = st.session_state.user_profile
    
    # Extract context from user input
    context = bot.extract_context(user_input)
    
    # Update profile with context
    if "energy" in context:
        profile.energy_level = context["energy"]
    if "time" in context:
        profile.available_time = context["time"]
    if "budget" in context:
        profile.budget_range = context["budget"]
    
    # Determine response type and generate appropriate response
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["workout", "exercise", "gym", "training"]):
        response = bot.generate_dynamic_workout(context, profile)
        response += "\n\n" + bot.track_streaks("workout", profile)
        
    elif any(word in user_lower for word in ["meal", "food", "eat", "nutrition", "diet"]):
        response = bot.generate_meal_suggestion(context, profile)
        response += "\n\n" + bot.track_streaks("nutrition", profile)
        
    elif any(word in user_lower for word in ["goal", "target", "achieve", "want to"]):
        response = bot.generate_smart_goals(user_input)
        
    elif any(word in user_lower for word in ["challenge", "motivate", "motivation"]):
        response = bot.generate_weekly_challenge()
        
    elif any(word in user_lower for word in ["water", "hydration", "drink"]):
        response = bot.hydration_intelligence(context)
        
    elif any(word in user_lower for word in ["tired", "stressed", "sad", "anxious", "angry", "excited"]):
        mood = next((word for word in ["tired", "stressed", "sad", "anxious", "angry", "excited"] if word in user_lower), "neutral")
        response = bot.mood_fitness_correlation(mood, profile)
        
    elif any(word in user_lower for word in ["recovery", "rest", "sore", "tired muscles"]):
        response = bot.recovery_optimization(context, profile)
        
    elif "reminder" in user_lower:
        response = bot.intelligent_reminders(profile)
        
    else:
        # Fallback to original chatbot rules
        response = "I can help with personalized workouts, meal planning, goal setting, challenges, hydration, mood-based fitness, and recovery! What interests you? 🤖"
    
    # Remember conversation
    bot.conversation_memory.append({
        'timestamp': datetime.datetime.now(),
        'user': user_input,
        'bot': response,
        'context': context
    })
    
    return response

# Original chatbot rules for fallback
chatbot_rules = {
    r".*\b(hi|hello|hey|greetings)\b.*": {
        'response': "Hello! 👋 I'm your enhanced fitness companion with personalized workouts, meal planning, and smart goal tracking!",
        'intent': 'greeting'
    },
    r".*\b(help)\b.*": {
        'response': "I can help with:\n🏋️ Dynamic workouts\n🍽️ Smart meal planning\n🎯 Goal setting\n💧 Hydration tracking\n😊 Mood-based fitness\n🏆 Weekly challenges\n⚡ Recovery optimization",
        'intent': 'help'
    },
    r".*\b(thank you|thanks)\b.*": {
        'response': "You're welcome! Keep crushing your fitness goals! 🙌",
        'intent': 'thank_you'
    },
    r".*\b(bye|goodbye|exit|quit)\b.*": {
        'response': "Goodbye! Stay consistent and remember - progress over perfection! 👋",
        'intent': 'exit'
    }
}

def get_fallback_response(user_input: str) -> str:
    """Fallback to original rule-based responses"""
    cleaned_input = re.sub(r'[^\w\s]', '', user_input.lower())
    
    for pattern, rule_data in chatbot_rules.items():
        if re.search(pattern, cleaned_input):
            return rule_data['response']
    
    return "I'm here to help with your fitness journey! Try asking about workouts, meals, goals, or challenges! 🤖"