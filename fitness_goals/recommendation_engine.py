"""
Recommendation engine for personalized fitness and nutrition advice.

This module provides the RecommendationEngine class that generates
goal-specific workout recommendations, nutrition advice, and motivational
messages based on user goals and progress trends.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

from .models import FitnessGoal, ProgressEntry


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Provides personalized fitness and nutrition recommendations based on user goals.
    
    Generates goal-specific advice for workouts, nutrition, and motivation
    tailored to weight loss, muscle building, or weight maintenance goals.
    """
    
    def __init__(self):
        """Initialize the recommendation engine with advice databases."""
        self._initialize_workout_recommendations()
        self._initialize_nutrition_recommendations()
        self._initialize_motivational_messages()
        logger.info("RecommendationEngine initialized")
    
    def get_workout_recommendations(self, goal: FitnessGoal, intent: str = "general") -> str:
        """
        Get workout recommendations based on goal type and specific intent.
        
        Args:
            goal: User's fitness goal
            intent: Specific workout intent (e.g., "cardio", "strength", "general")
            
        Returns:
            Personalized workout recommendation string
        """
        try:
            goal_type = goal.goal_type
            recommendations = self.workout_recommendations.get(goal_type, {})
            
            # Get specific intent recommendations or fall back to general
            if intent in recommendations:
                advice_list = recommendations[intent]
            else:
                advice_list = recommendations.get("general", [])
            
            if not advice_list:
                return self._get_fallback_workout_advice(goal_type)
            
            # Select random advice and personalize it
            base_advice = random.choice(advice_list)
            personalized_advice = self._personalize_workout_advice(base_advice, goal)
            
            logger.info(f"Generated workout recommendation for {goal_type} goal with intent '{intent}'")
            return personalized_advice
            
        except Exception as e:
            logger.error(f"Error generating workout recommendations: {str(e)}")
            return self._get_fallback_workout_advice(goal.goal_type)
    
    def get_nutrition_recommendations(self, goal: FitnessGoal, meal_type: str = "general") -> str:
        """
        Get nutrition recommendations based on goal type and meal type.
        
        Args:
            goal: User's fitness goal
            meal_type: Type of meal advice needed (e.g., "breakfast", "lunch", "dinner", "snack", "general")
            
        Returns:
            Personalized nutrition recommendation string
        """
        try:
            goal_type = goal.goal_type
            recommendations = self.nutrition_recommendations.get(goal_type, {})
            
            # Get specific meal type recommendations or fall back to general
            if meal_type in recommendations:
                advice_list = recommendations[meal_type]
            else:
                advice_list = recommendations.get("general", [])
            
            if not advice_list:
                return self._get_fallback_nutrition_advice(goal_type)
            
            # Select random advice and personalize it
            base_advice = random.choice(advice_list)
            personalized_advice = self._personalize_nutrition_advice(base_advice, goal)
            
            logger.info(f"Generated nutrition recommendation for {goal_type} goal with meal type '{meal_type}'")
            return personalized_advice
            
        except Exception as e:
            logger.error(f"Error generating nutrition recommendations: {str(e)}")
            return self._get_fallback_nutrition_advice(goal.goal_type)
    
    def get_motivational_message(self, goal: FitnessGoal, progress_trend: Dict[str, Any]) -> str:
        """
        Get motivational message based on goal and progress trend.
        
        Args:
            goal: User's fitness goal
            progress_trend: Dictionary containing progress analysis
                          Expected keys: 'status' ('on_track', 'ahead', 'behind'), 'progress_percentage'
            
        Returns:
            Personalized motivational message string
        """
        try:
            goal_type = goal.goal_type
            trend_status = progress_trend.get('status', 'unknown')
            progress_percentage = progress_trend.get('progress_percentage', 0)
            
            # Determine message category based on progress
            if trend_status == 'ahead':
                message_category = 'ahead'
            elif trend_status == 'behind':
                message_category = 'behind'
            elif trend_status == 'on_track':
                message_category = 'on_track'
            else:
                message_category = 'general'
            
            # Get messages for goal type and category
            messages = self.motivational_messages.get(goal_type, {}).get(message_category, [])
            
            if not messages:
                messages = self.motivational_messages.get('general', {}).get(message_category, [])
            
            if not messages:
                return self._get_fallback_motivational_message(goal_type, trend_status)
            
            # Select random message and personalize it
            base_message = random.choice(messages)
            personalized_message = self._personalize_motivational_message(base_message, goal, progress_trend)
            
            logger.info(f"Generated motivational message for {goal_type} goal with trend '{trend_status}'")
            return personalized_message
            
        except Exception as e:
            logger.error(f"Error generating motivational message: {str(e)}")
            return self._get_fallback_motivational_message(goal.goal_type, progress_trend.get('status', 'unknown'))
    
    def get_goal_specific_advice(self, goal: FitnessGoal, user_query: str) -> str:
        """
        Get goal-specific advice based on user query and goal type.
        
        Args:
            goal: User's fitness goal
            user_query: User's question or request
            
        Returns:
            Goal-specific advice string
        """
        try:
            query_lower = user_query.lower()
            goal_type = goal.goal_type
            
            # Determine intent from query
            if any(word in query_lower for word in ['workout', 'exercise', 'training', 'gym']):
                return self.get_workout_recommendations(goal, "general")
            elif any(word in query_lower for word in ['food', 'eat', 'meal', 'diet', 'nutrition']):
                return self.get_nutrition_recommendations(goal, "general")
            elif any(word in query_lower for word in ['progress', 'motivation', 'stuck', 'plateau']):
                # Create a neutral progress trend for general motivation
                neutral_trend = {'status': 'on_track', 'progress_percentage': 50}
                return self.get_motivational_message(goal, neutral_trend)
            else:
                # General goal-specific advice
                return self._get_general_goal_advice(goal)
            
        except Exception as e:
            logger.error(f"Error generating goal-specific advice: {str(e)}")
            return f"I understand you have a {goal.goal_type.replace('_', ' ')} goal. Let me help you with that!"
    
    def _initialize_workout_recommendations(self):
        """Initialize workout recommendation database."""
        self.workout_recommendations = {
            'weight_loss': {
                'general': [
                    "Focus on cardio exercises like running, cycling, or swimming for 30-45 minutes, 4-5 times per week. Add 2-3 strength training sessions to maintain muscle mass.",
                    "Try high-intensity interval training (HIIT) 3 times per week - alternate between 30 seconds of intense exercise and 90 seconds of rest for 20-30 minutes.",
                    "Combine steady-state cardio with circuit training. Do 20-30 minutes of moderate cardio followed by bodyweight circuits targeting all major muscle groups.",
                    "Walking is excellent for weight loss! Aim for 8,000-10,000 steps daily, plus 2-3 strength training sessions per week to preserve muscle."
                ],
                'cardio': [
                    "For effective fat burning, try 45-60 minutes of moderate-intensity cardio in the 'fat-burning zone' (60-70% max heart rate).",
                    "Mix up your cardio routine: alternate between running, cycling, rowing, and elliptical to prevent boredom and work different muscle groups.",
                    "Try incline walking on a treadmill - it's easier on joints but highly effective for burning calories and targeting glutes and legs."
                ],
                'strength': [
                    "Focus on compound movements like squats, deadlifts, and push-ups that work multiple muscle groups and burn more calories.",
                    "Use lighter weights with higher repetitions (12-15 reps) to maintain muscle while in a calorie deficit.",
                    "Include full-body strength training 2-3 times per week to preserve lean muscle mass during weight loss."
                ]
            },
            'muscle_building': {
                'general': [
                    "Focus on progressive overload with compound exercises like squats, deadlifts, bench press, and rows. Train each muscle group 2-3 times per week.",
                    "Prioritize heavy lifting with 6-8 reps per set for strength, and 8-12 reps for muscle growth. Rest 2-3 minutes between sets.",
                    "Follow a structured program like push/pull/legs or upper/lower split. Consistency and progressive overload are key for muscle growth.",
                    "Include both compound and isolation exercises. Start with big movements, then target specific muscles with focused exercises."
                ],
                'strength': [
                    "Focus on the 'big three': squat, bench press, and deadlift. These compound movements build overall strength and muscle mass.",
                    "Use the 5x5 method: 5 sets of 5 reps with heavy weight, focusing on perfect form and progressive overload.",
                    "Train in the 1-6 rep range with 85-95% of your max weight to build maximum strength and dense muscle."
                ],
                'hypertrophy': [
                    "Train in the 8-12 rep range with moderate weight for optimal muscle growth. Focus on time under tension.",
                    "Use drop sets, supersets, and other intensity techniques to maximize muscle stimulation and growth.",
                    "Target each muscle group with 12-20 sets per week, spread across 2-3 training sessions for optimal recovery."
                ]
            },
            'weight_maintenance': {
                'general': [
                    "Maintain a balanced routine with 3-4 days of mixed cardio and strength training. Focus on activities you enjoy for long-term adherence.",
                    "Try a 3-day full-body strength routine plus 2-3 days of moderate cardio like walking, swimming, or cycling.",
                    "Focus on functional fitness: exercises that improve daily life activities like squats, lunges, push-ups, and core work.",
                    "Vary your routine seasonally - outdoor activities in summer, gym workouts in winter to keep things interesting."
                ],
                'cardio': [
                    "Aim for 150 minutes of moderate cardio per week, as recommended by health guidelines. This can be 30 minutes, 5 days a week.",
                    "Mix low and moderate intensity activities: walking, hiking, dancing, swimming - choose what you enjoy most.",
                    "Try active hobbies like tennis, basketball, or cycling with friends to make cardio more social and fun."
                ],
                'strength': [
                    "Maintain muscle mass with 2-3 strength sessions per week, focusing on all major muscle groups.",
                    "Use bodyweight exercises or light weights with higher reps to maintain strength without excessive muscle growth.",
                    "Focus on functional movements that support daily activities and prevent age-related muscle loss."
                ]
            }
        }
    
    def _initialize_nutrition_recommendations(self):
        """Initialize nutrition recommendation database."""
        self.nutrition_recommendations = {
            'weight_loss': {
                'general': [
                    "Create a moderate calorie deficit of 300-500 calories per day. Focus on whole foods: lean proteins, vegetables, fruits, and whole grains.",
                    "Prioritize protein (0.8-1g per lb body weight) to preserve muscle mass. Include protein at every meal and snack.",
                    "Fill half your plate with non-starchy vegetables, one quarter with lean protein, and one quarter with complex carbs.",
                    "Stay hydrated and consider eating smaller, more frequent meals to help control hunger and maintain energy levels."
                ],
                'breakfast': [
                    "Start with protein: eggs, Greek yogurt, or protein smoothies. Add fiber from oats, berries, or vegetables.",
                    "Try overnight oats with protein powder, berries, and nuts for a filling, nutritious start to your day.",
                    "Vegetable omelets with a side of fruit provide protein, fiber, and essential nutrients while keeping calories controlled."
                ],
                'lunch': [
                    "Build salads with lean protein (chicken, fish, tofu), lots of vegetables, and a small amount of healthy fats like avocado or nuts.",
                    "Try soup and salad combinations - broth-based soups are filling and low in calories when paired with a side salad.",
                    "Wrap or bowl meals with plenty of vegetables, lean protein, and a small portion of whole grains."
                ],
                'dinner': [
                    "Keep dinner lighter with grilled or baked protein, steamed vegetables, and a small portion of complex carbs.",
                    "Try zucchini noodles or cauliflower rice as lower-calorie alternatives to pasta and rice.",
                    "Focus on lean proteins like fish, chicken breast, or plant-based options with plenty of colorful vegetables."
                ],
                'snack': [
                    "Choose protein-rich snacks: Greek yogurt, hard-boiled eggs, or a small handful of nuts with fruit.",
                    "Vegetables with hummus, apple slices with almond butter, or cottage cheese with berries are satisfying options.",
                    "Keep portions controlled and focus on snacks that provide protein and fiber to help you feel full."
                ]
            },
            'muscle_building': {
                'general': [
                    "Eat in a slight calorie surplus (200-500 calories above maintenance) with emphasis on protein (1-1.2g per lb body weight).",
                    "Time your nutrition around workouts: eat carbs and protein before and after training for optimal muscle protein synthesis.",
                    "Include healthy fats (20-30% of calories) from sources like nuts, avocado, olive oil, and fatty fish for hormone production.",
                    "Eat frequently throughout the day to maintain positive nitrogen balance and support muscle growth."
                ],
                'breakfast': [
                    "Start big with eggs, oats, and fruit. Consider adding protein powder to oatmeal or making protein pancakes.",
                    "Try a muscle-building smoothie with protein powder, banana, oats, peanut butter, and milk.",
                    "Whole grain toast with avocado and eggs provides healthy fats, protein, and complex carbs for sustained energy."
                ],
                'lunch': [
                    "Focus on substantial meals with lean meats, complex carbs like rice or quinoa, and plenty of vegetables.",
                    "Try chicken and rice bowls with vegetables, or lean beef with sweet potato and broccoli.",
                    "Include healthy fats like nuts, seeds, or olive oil to increase calorie density without excessive volume."
                ],
                'dinner': [
                    "Make dinner your largest meal with plenty of protein, complex carbs, and vegetables for recovery.",
                    "Salmon with quinoa and roasted vegetables provides protein, healthy fats, and nutrients for muscle growth.",
                    "Consider casein protein or Greek yogurt before bed to support overnight muscle protein synthesis."
                ],
                'snack': [
                    "Focus on protein-rich snacks between meals: protein shakes, trail mix, or chocolate milk post-workout.",
                    "Peanut butter and banana sandwiches, or Greek yogurt with granola provide calories and protein.",
                    "Don't skip snacks - they help you reach your calorie and protein goals for muscle growth."
                ]
            },
            'weight_maintenance': {
                'general': [
                    "Focus on balanced nutrition with adequate protein, healthy fats, and complex carbs. Listen to hunger and fullness cues.",
                    "Follow the 80/20 rule: eat nutritiously 80% of the time, allow flexibility for treats and social eating 20% of the time.",
                    "Maintain consistent meal timing and include all food groups for optimal health and energy levels.",
                    "Stay hydrated and focus on whole, minimally processed foods while allowing room for foods you enjoy."
                ],
                'breakfast': [
                    "Aim for balanced breakfasts with protein, healthy fats, and complex carbs to start your day right.",
                    "Oatmeal with nuts and fruit, or eggs with whole grain toast provide sustained energy and satisfaction.",
                    "Include variety in your breakfast routine to ensure you get different nutrients throughout the week."
                ],
                'lunch': [
                    "Build balanced meals with a palm-sized portion of protein, fist-sized portion of vegetables, and cupped-hand portion of carbs.",
                    "Try Buddha bowls with various vegetables, protein, healthy fats, and whole grains for complete nutrition.",
                    "Focus on meals that provide steady energy for your afternoon activities and workouts."
                ],
                'dinner': [
                    "Keep dinners balanced and satisfying without being overly heavy, especially if you eat late.",
                    "Include a variety of colors on your plate to ensure you're getting diverse nutrients and antioxidants.",
                    "Practice mindful eating - eat slowly and pay attention to hunger and fullness signals."
                ],
                'snack': [
                    "Choose snacks based on your activity level and hunger - don't snack out of boredom or habit.",
                    "Combine protein with carbs or healthy fats for sustained energy: apple with almond butter, or yogurt with berries.",
                    "Keep healthy snacks available to avoid reaching for less nutritious options when hunger strikes."
                ]
            }
        }
    
    def _initialize_motivational_messages(self):
        """Initialize motivational message database."""
        self.motivational_messages = {
            'weight_loss': {
                'ahead': [
                    "Fantastic progress! You're ahead of schedule on your weight loss journey. Keep up the excellent work!",
                    "You're crushing your weight loss goals! Your dedication is really paying off. Stay consistent!",
                    "Amazing job! You're exceeding expectations. Remember to celebrate these victories along the way!"
                ],
                'on_track': [
                    "You're right on track with your weight loss goal! Consistency is key, and you're nailing it.",
                    "Perfect pace! You're making steady progress toward your target weight. Keep doing what you're doing!",
                    "Great job staying consistent! You're on the right path to reaching your weight loss goal."
                ],
                'behind': [
                    "Don't get discouraged! Weight loss isn't always linear. Focus on your healthy habits and trust the process.",
                    "Every small step counts! Even if progress seems slow, you're building lasting healthy habits.",
                    "Remember why you started. Small setbacks are normal - what matters is getting back on track!"
                ],
                'general': [
                    "Every healthy choice you make is an investment in your future self. You've got this!",
                    "Weight loss is a journey, not a destination. Focus on building sustainable habits.",
                    "Be patient with yourself. Lasting change takes time, but you're making progress every day!"
                ]
            },
            'muscle_building': {
                'ahead': [
                    "Incredible gains! You're building muscle faster than expected. Your hard work in the gym is paying off!",
                    "Outstanding progress! You're ahead of schedule on your muscle building journey. Keep lifting!",
                    "Your dedication to training and nutrition is showing! You're exceeding your muscle building goals!"
                ],
                'on_track': [
                    "Perfect progress! You're right on track with your muscle building goals. Consistency wins!",
                    "Great job! You're making steady gains and building quality muscle. Keep up the excellent work!",
                    "You're hitting your targets! Your commitment to progressive overload is paying off."
                ],
                'behind': [
                    "Muscle building takes time - don't rush the process! Focus on progressive overload and proper nutrition.",
                    "Remember, quality gains take time. Make sure you're eating enough and getting adequate rest for recovery.",
                    "Plateaus are normal in muscle building. Consider adjusting your program or increasing your calorie intake."
                ],
                'general': [
                    "Building muscle is a marathon, not a sprint. Stay consistent with your training and nutrition!",
                    "Every rep counts! You're getting stronger with each workout, even when you can't see it yet.",
                    "Trust the process! Muscle growth happens gradually, but your efforts will compound over time."
                ]
            },
            'weight_maintenance': {
                'ahead': [
                    "Excellent job maintaining your weight! You've found a great balance with your lifestyle.",
                    "Perfect maintenance! You're successfully balancing your nutrition and activity levels.",
                    "Great work! You've achieved a sustainable balance that works for your lifestyle."
                ],
                'on_track': [
                    "You're maintaining beautifully! Your balanced approach to fitness and nutrition is working perfectly.",
                    "Fantastic maintenance! You've found your sweet spot for long-term health and wellness.",
                    "Perfect balance! You're successfully maintaining your weight while enjoying life."
                ],
                'behind': [
                    "Small fluctuations are completely normal! Focus on your overall healthy habits and patterns.",
                    "Don't worry about minor changes - weight naturally fluctuates. Keep up your healthy routine!",
                    "Remember, maintenance is about long-term patterns, not daily numbers. You're doing great!"
                ],
                'general': [
                    "Maintenance is a skill! You're successfully balancing health, fitness, and enjoying life.",
                    "You've achieved something many struggle with - sustainable weight maintenance. Well done!",
                    "Your balanced approach to health and fitness is inspiring. Keep up the great work!"
                ]
            },
            'general': {
                'ahead': [
                    "Outstanding progress! You're exceeding expectations and should be proud of your dedication!",
                    "Incredible work! You're ahead of schedule and showing what consistency can achieve!",
                    "Amazing job! Your hard work and dedication are really paying off!"
                ],
                'on_track': [
                    "Perfect progress! You're right where you need to be. Keep up the excellent work!",
                    "Great job staying consistent! You're making steady progress toward your goals.",
                    "You're doing everything right! Stay the course and trust the process."
                ],
                'behind': [
                    "Don't give up! Progress isn't always linear, but every effort counts toward your goal.",
                    "Remember, setbacks are part of the journey. What matters is getting back on track!",
                    "Be patient with yourself. Lasting change takes time, but you're moving in the right direction!"
                ],
                'general': [
                    "Every step forward is progress! You're building healthy habits that will last a lifetime.",
                    "Your commitment to your health and fitness goals is admirable. Keep going!",
                    "Remember why you started and how far you've come. You're stronger than you think!"
                ]
            }
        }
    
    def _personalize_workout_advice(self, base_advice: str, goal: FitnessGoal) -> str:
        """Personalize workout advice with goal-specific details."""
        try:
            # Add goal-specific context
            if goal.goal_type == 'weight_loss':
                target_loss = goal.current_value - goal.target_value
                weeks_remaining = goal.timeframe_weeks
                context = f" With {target_loss:.1f}kg to lose in {weeks_remaining} weeks, "
            elif goal.goal_type == 'muscle_building':
                target_gain = goal.target_value - goal.current_value
                weeks_remaining = goal.timeframe_weeks
                context = f" To gain {target_gain:.1f}kg of muscle in {weeks_remaining} weeks, "
            else:
                context = f" For maintaining your current weight of {goal.current_value:.1f}kg, "
            
            return context + base_advice.lower()
            
        except Exception:
            return base_advice
    
    def _personalize_nutrition_advice(self, base_advice: str, goal: FitnessGoal) -> str:
        """Personalize nutrition advice with goal-specific details."""
        try:
            # Add goal-specific context
            if goal.goal_type == 'weight_loss':
                context = f"For your weight loss goal (current: {goal.current_value:.1f}kg, target: {goal.target_value:.1f}kg), "
            elif goal.goal_type == 'muscle_building':
                context = f"To support muscle growth (current: {goal.current_value:.1f}kg, target: {goal.target_value:.1f}kg), "
            else:
                context = f"For maintaining your weight at {goal.current_value:.1f}kg, "
            
            return context + base_advice.lower()
            
        except Exception:
            return base_advice
    
    def _personalize_motivational_message(self, base_message: str, goal: FitnessGoal, progress_trend: Dict[str, Any]) -> str:
        """Personalize motivational message with specific progress details."""
        try:
            progress_percentage = progress_trend.get('progress_percentage', 0)
            days_remaining = goal.days_remaining()
            
            # Add specific progress context
            if progress_percentage > 0:
                progress_context = f" You're {progress_percentage:.0f}% of the way to your goal"
                if days_remaining > 0:
                    progress_context += f" with {days_remaining} days remaining"
                progress_context += ". "
            else:
                progress_context = " "
            
            return base_message + progress_context
            
        except Exception:
            return base_message
    
    def _get_general_goal_advice(self, goal: FitnessGoal) -> str:
        """Get general advice based on goal type."""
        goal_type = goal.goal_type.replace('_', ' ')
        
        if goal.goal_type == 'weight_loss':
            return f"For your {goal_type} goal, focus on creating a sustainable calorie deficit through a combination of cardio exercise and strength training, while maintaining a balanced diet rich in protein and vegetables."
        elif goal.goal_type == 'muscle_building':
            return f"For your {goal_type} goal, prioritize progressive overload in your strength training, ensure adequate protein intake (1-1.2g per lb body weight), and maintain a slight calorie surplus to support muscle growth."
        else:
            return f"For your {goal_type} goal, focus on maintaining a balanced routine of cardio and strength training, while eating intuitively and listening to your body's hunger and fullness cues."
    
    def _get_fallback_workout_advice(self, goal_type: str) -> str:
        """Get fallback workout advice when specific recommendations aren't available."""
        if goal_type == 'weight_loss':
            return "For weight loss, combine cardio exercises with strength training. Aim for 150-300 minutes of moderate cardio per week plus 2-3 strength sessions."
        elif goal_type == 'muscle_building':
            return "For muscle building, focus on progressive overload with compound exercises. Train each muscle group 2-3 times per week with adequate rest between sessions."
        else:
            return "For weight maintenance, aim for a balanced routine of 150 minutes moderate cardio per week plus 2-3 strength training sessions."
    
    def _get_fallback_nutrition_advice(self, goal_type: str) -> str:
        """Get fallback nutrition advice when specific recommendations aren't available."""
        if goal_type == 'weight_loss':
            return "For weight loss, create a moderate calorie deficit while prioritizing protein, vegetables, and whole grains. Stay hydrated and eat regular meals."
        elif goal_type == 'muscle_building':
            return "For muscle building, eat in a slight calorie surplus with emphasis on protein (1-1.2g per lb body weight). Include healthy fats and complex carbs."
        else:
            return "For weight maintenance, focus on balanced nutrition with all food groups. Listen to hunger cues and maintain consistent eating patterns."
    
    def _get_fallback_motivational_message(self, goal_type: str, trend_status: str) -> str:
        """Get fallback motivational message when specific messages aren't available."""
        goal_name = goal_type.replace('_', ' ')
        
        if trend_status == 'ahead':
            return f"Great job! You're making excellent progress on your {goal_name} goal!"
        elif trend_status == 'behind':
            return f"Stay positive! Progress on your {goal_name} goal takes time and consistency."
        else:
            return f"Keep up the good work on your {goal_name} goal! Consistency is key to success."


# Global instance for easy access
_recommendation_engine_instance = None


def get_recommendation_engine() -> RecommendationEngine:
    """
    Get the global RecommendationEngine instance (singleton pattern).
    
    Returns:
        RecommendationEngine instance
    """
    global _recommendation_engine_instance
    if _recommendation_engine_instance is None:
        _recommendation_engine_instance = RecommendationEngine()
    return _recommendation_engine_instance