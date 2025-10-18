"""
Enhanced chatbot functionality with goal awareness and proactive suggestions.

This module extends the existing chatbot with goal-aware responses,
proactive goal-setting suggestions, and personalized advice based on user goals.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import streamlit as st

from .models import FitnessGoal
from .goal_manager import GoalManager, GoalManagerError
from .recommendation_engine import RecommendationEngine
from .data_storage import DataStorage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedChatbot:
    """
    Enhanced chatbot with goal awareness and personalized responses.
    
    Integrates with the existing rule-based chatbot to provide goal-specific
    advice, proactive goal suggestions, and milestone tracking.
    """
    
    def __init__(self, goal_manager: Optional[GoalManager] = None, 
                 recommendation_engine: Optional[RecommendationEngine] = None):
        """
        Initialize enhanced chatbot with goal management and recommendation capabilities.
        
        Args:
            goal_manager: GoalManager instance for goal operations
            recommendation_engine: RecommendationEngine for personalized advice
        """
        self.goal_manager = goal_manager or GoalManager()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self._initialize_conversation_tracking()
        logger.info("EnhancedChatbot initialized")
    
    def _initialize_conversation_tracking(self):
        """Initialize conversation tracking in session state."""
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
        
        if 'last_goal_suggestion' not in st.session_state:
            st.session_state.last_goal_suggestion = None
        
        if 'goal_suggestion_declined' not in st.session_state:
            st.session_state.goal_suggestion_declined = False
        
        if 'goal_decline_count' not in st.session_state:
            st.session_state.goal_decline_count = 0
        
        if 'fitness_objectives_mentioned' not in st.session_state:
            st.session_state.fitness_objectives_mentioned = []
        
        if 'goal_suggestion_triggers' not in st.session_state:
            st.session_state.goal_suggestion_triggers = []
        
        if 'last_decline_timestamp' not in st.session_state:
            st.session_state.last_decline_timestamp = None
    
    def get_enhanced_response(self, user_input: str, original_response: str, 
                            user_id: str, intent: str = None) -> str:
        """
        Get enhanced response with goal awareness and personalization.
        
        Args:
            user_input: User's input message
            original_response: Original chatbot response
            user_id: User identifier
            intent: Detected intent from original chatbot
            
        Returns:
            Enhanced response with goal-specific context
        """
        try:
            # Increment conversation count
            current_count = st.session_state.get('conversation_count', 0)
            st.session_state.conversation_count = current_count + 1
            
            # Get user's current goal
            user_goal = self._get_user_goal(user_id)
            
            # Check for fitness objectives in user input first (Requirement 4.5)
            fitness_objective = self._detect_fitness_objectives(user_input)
            
            # If user mentions fitness objectives, always suggest goal setting
            if not user_goal and fitness_objective:
                # Don't detect again in the suggestion response since we already detected it
                return self._generate_goal_suggestion_response(user_input, original_response, fitness_objective)
            
            # Check if we should suggest goal setting for new users
            if not user_goal and self._should_suggest_goal_setting():
                return self._generate_goal_suggestion_response(user_input, original_response)
            
            # If user has a goal, enhance the response with goal-specific advice
            if user_goal:
                enhanced_response = self._enhance_response_with_goal_context(
                    user_input, original_response, user_goal, intent
                )
                
                # Check for goal milestones and add celebration messages
                milestone_message = self._check_goal_milestones(user_goal, user_id)
                if milestone_message:
                    enhanced_response += f"\n\n{milestone_message}"
                
                return enhanced_response
            
            # For users without goals, occasionally remind about goal setting
            if self._should_remind_about_goals():
                return original_response + "\n\n💡 *Tip: Setting a specific fitness goal can help me provide more personalized advice! Would you like to set a goal?*"
            
            return original_response
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {str(e)}")
            return original_response
    
    def _get_user_goal(self, user_id: str) -> Optional[FitnessGoal]:
        """Get user's current goal safely."""
        try:
            return self.goal_manager.get_user_goal(user_id)
        except GoalManagerError as e:
            logger.warning(f"Error retrieving goal for user {user_id}: {str(e)}")
            return None
    
    def _should_suggest_goal_setting(self) -> bool:
        """
        Determine if we should proactively suggest goal setting.
        
        Returns:
            True if goal setting should be suggested
        """
        conversation_count = st.session_state.get('conversation_count', 0)
        
        # Requirement 4.2: Ask about fitness objectives within first 3 interactions
        if conversation_count <= 3:
            return True
        
        # Check if user has declined recently and apply backoff strategy
        if st.session_state.get('goal_suggestion_declined', False):
            decline_count = st.session_state.get('goal_decline_count', 0)
            last_decline = st.session_state.get('last_decline_timestamp')
            
            # Requirement 4.4: Periodically suggest goal setting after decline
            if last_decline:
                # Calculate time-based backoff (in conversations)
                min_conversations_since_decline = 10 + (decline_count * 5)
                decline_conversation = last_decline.get('conversation_count', 0)
                conversations_since_decline = conversation_count - decline_conversation
                
                if conversations_since_decline < min_conversations_since_decline:
                    return False
                else:
                    # Enough time has passed, reset decline flag and allow suggestion
                    st.session_state.goal_suggestion_declined = False
                    return True
            
            # Reset decline flag after sufficient time has passed (alternative path)
            if decline_count <= 2 and conversation_count % (20 + decline_count * 10) == 0:
                st.session_state.goal_suggestion_declined = False
                return True
        
        # Check for fitness objective triggers
        fitness_triggers = st.session_state.get('goal_suggestion_triggers', [])
        if fitness_triggers:
            # Suggest if user mentioned fitness objectives recently
            recent_triggers = [t for t in fitness_triggers if conversation_count - t.get('conversation', 0) <= 2]
            if recent_triggers:
                return True
        
        # Periodic suggestions for users without goals (less frequent after declines)
        decline_count = st.session_state.get('goal_decline_count', 0)
        suggestion_interval = 15 + (decline_count * 10)  # Increase interval after declines
        
        if conversation_count > 5 and conversation_count % suggestion_interval == 0:
            return True
        
        return False
    
    def _should_remind_about_goals(self) -> bool:
        """
        Determine if we should remind about goal setting benefits.
        
        Returns:
            True if reminder should be shown
        """
        # Remind every 20 conversations for users without goals
        conversation_count = st.session_state.get('conversation_count', 0)
        return (conversation_count > 5 and conversation_count % 20 == 0)
    
    def _generate_goal_suggestion_response(self, user_input: str, original_response: str, 
                                          pre_detected_objective: str = None) -> str:
        """
        Generate response that includes goal setting suggestion.
        
        Args:
            user_input: User's input message
            original_response: Original chatbot response
            
        Returns:
            Response with goal setting suggestion
        """
        conversation_count = st.session_state.get('conversation_count', 0)
        decline_count = st.session_state.get('goal_decline_count', 0)
        
        # Check if user input indicates fitness objectives (Requirement 4.5)
        fitness_indicators = pre_detected_objective or self._detect_fitness_objectives(user_input)
        
        if fitness_indicators:
            # Requirement 4.5: Offer to help formalize fitness objectives as trackable goals
            if fitness_indicators == "weight loss goals":
                suggestion = (
                    f"\n\n🎯 I noticed you mentioned weight loss! "
                    "Would you like me to help you set a specific weight loss goal? "
                    "I can track your progress and provide personalized meal and workout recommendations!"
                )
            elif fitness_indicators == "muscle building goals":
                suggestion = (
                    f"\n\n💪 I see you're interested in building muscle! "
                    "Want to set a specific muscle building goal? "
                    "I can help you track progress and suggest the best strength training routines!"
                )
            elif fitness_indicators == "maintenance goals":
                suggestion = (
                    f"\n\n⚖️ Maintaining your current fitness is a great goal! "
                    "Would you like to set up a maintenance plan? "
                    "I can help you stay consistent with balanced workouts and nutrition!"
                )
            else:
                suggestion = (
                    f"\n\n🎯 I noticed you mentioned {fitness_indicators}! "
                    "Would you like me to help you set a specific fitness goal? "
                    "I can provide much more personalized advice when I know your target!"
                )
        else:
            # Generate contextual suggestions based on conversation stage and decline history
            if conversation_count <= 3:
                # Early conversation suggestions (Requirement 4.2)
                suggestions = [
                    "\n\n🎯 **Welcome!** To give you the best fitness advice, would you like to set a specific goal? I can help with weight loss, muscle building, or maintenance!",
                    "\n\n💪 **Getting started?** Setting a fitness goal helps me provide personalized recommendations. What would you like to work toward?",
                    "\n\n🏆 **Ready to begin your fitness journey?** I'd love to help you set a goal so I can tailor my advice to your specific needs!"
                ]
            elif decline_count == 0:
                # First time suggesting after initial conversations
                suggestions = [
                    "\n\n🎯 *Would you like to set a fitness goal? I can provide much more personalized advice when I know what you're working toward!*",
                    "\n\n💪 *Setting a specific goal like weight loss, muscle building, or maintenance helps me give you better recommendations. Interested?*",
                    "\n\n🏆 *I'd love to help you create a personalized fitness plan! Would you like to set a goal so I can tailor my advice to your needs?*"
                ]
            else:
                # Gentler suggestions after previous declines (Requirement 4.4)
                suggestions = [
                    "\n\n💡 *Just a gentle reminder: I can provide more targeted advice if you ever want to set a fitness goal. No pressure though!*",
                    "\n\n🌟 *Whenever you're ready to set a specific fitness target, I'm here to help make it happen!*",
                    "\n\n✨ *If you ever want personalized fitness guidance, setting a goal really helps me give better recommendations!*"
                ]
            
            # Rotate through different suggestion styles based on conversation count
            suggestion_index = conversation_count % len(suggestions)
            suggestion = suggestions[suggestion_index]
        
        # Mark that we've made a suggestion
        st.session_state.last_goal_suggestion = {
            'conversation': conversation_count,
            'timestamp': datetime.now().isoformat(),
            'type': 'fitness_objective' if fitness_indicators else 'general'
        }
        
        return original_response + suggestion
    
    def _detect_fitness_objectives(self, user_input: str) -> str:
        """
        Detect fitness objectives mentioned in user input.
        
        Args:
            user_input: User's input message
            
        Returns:
            Detected objective description or empty string
        """
        input_lower = user_input.lower()
        detected_objective = ""
        
        # Weight loss indicators (more comprehensive)
        weight_loss_terms = [
            'lose weight', 'weight loss', 'get lean', 'cut fat', 'slim down', 'shed pounds',
            'drop pounds', 'burn fat', 'lose belly fat', 'get skinny', 'diet plan',
            'calorie deficit', 'want to be lighter', 'need to lose', 'trying to lose',
            'shed some pounds', 'need to shed'
        ]
        for term in weight_loss_terms:
            if term in input_lower:
                detected_objective = "weight loss goals"
                break
        
        # Muscle building indicators (more comprehensive)
        if not detected_objective:
            muscle_terms = [
                'build muscle', 'gain muscle', 'get bigger', 'bulk up', 'muscle mass', 'get stronger',
                'gain weight', 'put on muscle', 'muscle building', 'strength training',
                'want to be stronger', 'build mass', 'get buff', 'get ripped', 'tone up'
            ]
            for term in muscle_terms:
                if term in input_lower:
                    detected_objective = "muscle building goals"
                    break
        
        # Maintenance indicators (more comprehensive)
        if not detected_objective:
            maintenance_terms = [
                'maintain weight', 'maintain my weight', 'stay fit', 'keep current', 'maintain muscle',
                'stay in shape', 'keep my fitness', 'maintain current', 'stay healthy',
                'keep my body', 'maintain my fitness level', 'keep my current fitness level',
                'maintain my muscle'
            ]
            for term in maintenance_terms:
                if term in input_lower:
                    detected_objective = "maintenance goals"
                    break
        
        # General fitness indicators (more comprehensive)
        if not detected_objective:
            general_terms = [
                'get fit', 'fitness goal', 'workout plan', 'exercise routine', 'fitness journey',
                'start working out', 'get in shape', 'improve fitness', 'health goals',
                'fitness target', 'body goals', 'transformation', 'get healthy',
                'improve my fitness'
            ]
            for term in general_terms:
                if term in input_lower:
                    detected_objective = "fitness objectives"
                    break
        
        # Track detected objectives for future reference
        if detected_objective:
            self._track_fitness_objective_mention(detected_objective, user_input)
        
        return detected_objective
    
    def _track_fitness_objective_mention(self, objective_type: str, user_input: str):
        """
        Track when user mentions fitness objectives for proactive suggestions.
        
        Args:
            objective_type: Type of objective detected
            user_input: Original user input
        """
        conversation_count = st.session_state.get('conversation_count', 0)
        
        # Add to fitness objectives mentioned list
        objectives_mentioned = st.session_state.get('fitness_objectives_mentioned', [])
        objectives_mentioned.append({
            'type': objective_type,
            'input': user_input[:100],  # Store first 100 chars for context
            'conversation': conversation_count,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent mentions (last 10)
        if len(objectives_mentioned) > 10:
            objectives_mentioned = objectives_mentioned[-10:]
        
        st.session_state.fitness_objectives_mentioned = objectives_mentioned
        
        # Add to goal suggestion triggers
        triggers = st.session_state.get('goal_suggestion_triggers', [])
        triggers.append({
            'type': 'fitness_objective_detected',
            'objective': objective_type,
            'conversation': conversation_count,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent triggers (last 5)
        if len(triggers) > 5:
            triggers = triggers[-5:]
        
        st.session_state.goal_suggestion_triggers = triggers
    
    def _enhance_response_with_goal_context(self, user_input: str, original_response: str, 
                                          goal: FitnessGoal, intent: str) -> str:
        """
        Enhance response with goal-specific context and advice.
        
        Args:
            user_input: User's input message
            original_response: Original chatbot response
            goal: User's fitness goal
            intent: Detected intent from original chatbot
            
        Returns:
            Enhanced response with goal context
        """
        try:
            # Get goal-specific advice based on intent
            goal_advice = self._get_goal_specific_advice(user_input, goal, intent)
            
            if goal_advice and goal_advice != original_response:
                # Combine original response with goal-specific advice
                enhanced_response = f"{original_response}\n\n🎯 **For your {goal.goal_type.replace('_', ' ')} goal:** {goal_advice}"
            else:
                enhanced_response = original_response
            
            # Add goal progress context if relevant
            progress_context = self._get_progress_context(goal)
            if progress_context:
                enhanced_response += f"\n\n{progress_context}"
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response with goal context: {str(e)}")
            return original_response
    
    def _get_goal_specific_advice(self, user_input: str, goal: FitnessGoal, intent: str) -> str:
        """
        Get goal-specific advice based on user input and intent.
        
        Args:
            user_input: User's input message
            goal: User's fitness goal
            intent: Detected intent from original chatbot
            
        Returns:
            Goal-specific advice string
        """
        try:
            # Map intents to recommendation types
            if intent in ['workout_plan', 'beginner_workout', 'strength_workout', 'cardio_workout']:
                return self.recommendation_engine.get_workout_recommendations(goal, intent.replace('_workout', ''))
            
            elif intent in ['nutrition_plan', 'breakfast_ideas', 'lunch_ideas', 'dinner_ideas', 'snack_ideas']:
                meal_type = intent.replace('_ideas', '').replace('_plan', '')
                if meal_type == 'nutrition':
                    meal_type = 'general'
                return self.recommendation_engine.get_nutrition_recommendations(goal, meal_type)
            
            elif intent in ['motivation', 'app_features']:
                # Provide motivational message with neutral progress
                neutral_progress = {'status': 'on_track', 'progress_percentage': 50}
                return self.recommendation_engine.get_motivational_message(goal, neutral_progress)
            
            else:
                # For other intents, provide general goal-specific advice
                return self.recommendation_engine.get_goal_specific_advice(goal, user_input)
            
        except Exception as e:
            logger.error(f"Error getting goal-specific advice: {str(e)}")
            return ""
    
    def _get_progress_context(self, goal: FitnessGoal) -> str:
        """
        Get progress context message for the user's goal.
        
        Args:
            goal: User's fitness goal
            
        Returns:
            Progress context message or empty string
        """
        try:
            days_remaining = goal.days_remaining()
            
            if goal.is_overdue():
                return f"⏰ *Your goal deadline has passed. Consider updating your timeline or setting a new goal!*"
            elif days_remaining <= 7:
                return f"⏰ *You have {days_remaining} days left to reach your goal! Stay focused!*"
            elif days_remaining <= 30:
                return f"📅 *{days_remaining} days remaining to reach your goal. You're in the home stretch!*"
            else:
                weeks_remaining = days_remaining // 7
                return f"📊 *{weeks_remaining} weeks remaining in your {goal.goal_type.replace('_', ' ')} journey.*"
            
        except Exception as e:
            logger.error(f"Error getting progress context: {str(e)}")
            return ""
    
    def _check_goal_milestones(self, goal: FitnessGoal, user_id: str) -> str:
        """
        Check for goal milestones and return celebration messages.
        
        Args:
            goal: User's fitness goal
            user_id: User identifier
            
        Returns:
            Milestone celebration message or empty string
        """
        try:
            # Calculate time-based milestones
            total_days = (goal.target_date - goal.created_date).days
            days_elapsed = total_days - goal.days_remaining()
            time_progress = (days_elapsed / total_days) if total_days > 0 else 0
            
            # Check for milestone celebrations (25%, 50%, 75% time elapsed)
            milestones = [0.25, 0.5, 0.75]
            milestone_key = f"milestone_{goal.goal_id}"
            
            celebrated_milestones = st.session_state.get(milestone_key, [])
            
            for milestone in milestones:
                if time_progress >= milestone and milestone not in celebrated_milestones:
                    celebrated_milestones.append(milestone)
                    st.session_state[milestone_key] = celebrated_milestones
                    
                    percentage = int(milestone * 100)
                    return f"🎉 **Milestone Alert!** You're {percentage}% through your {goal.goal_type.replace('_', ' ')} timeline! Keep up the great work!"
            
            return ""
            
        except Exception as e:
            logger.error(f"Error checking goal milestones: {str(e)}")
            return ""
    
    def handle_goal_response(self, user_input: str, user_id: str) -> Tuple[bool, str]:
        """
        Handle user responses to goal-setting suggestions.
        
        Args:
            user_input: User's response
            user_id: User identifier
            
        Returns:
            Tuple of (handled, response_message)
        """
        input_lower = user_input.lower().strip()
        conversation_count = st.session_state.get('conversation_count', 0)
        
        # Check if this is a response to a recent goal suggestion
        last_suggestion = st.session_state.get('last_goal_suggestion')
        if not last_suggestion:
            return False, ""
        
        # Only handle responses within 2 conversations of the suggestion
        if conversation_count - last_suggestion.get('conversation', 0) > 2:
            return False, ""
        
        # Ambiguous responses - provide clarification (check first to avoid conflicts)
        # Use exact word matching to avoid conflicts with "maybe later"
        ambiguous_responses = ['maybe', 'i don\'t know', 'not sure', 'what do you think']
        is_ambiguous = False
        for response in ambiguous_responses:
            if response == 'maybe':
                # Check for exact "maybe" but not "maybe later"
                if ' maybe ' in f' {input_lower} ' and 'later' not in input_lower:
                    is_ambiguous = True
                    break
                elif input_lower == 'maybe':
                    is_ambiguous = True
                    break
            else:
                if response in input_lower:
                    is_ambiguous = True
                    break
        
        if is_ambiguous:
            return True, (
                "No worries! Setting a goal is completely optional. "
                "I can help you either way - with general fitness advice or personalized guidance if you set a goal. "
                "What would you prefer? 😊"
            )
        
        # Positive responses to goal setting (Requirement 4.3)
        positive_responses = [
            'yes', 'yeah', 'sure', 'ok', 'okay', 'let\'s do it', 'help me set a goal',
            'i want to', 'sounds good', 'let\'s go', 'i\'m interested', 'tell me more',
            'how do i', 'what do i need', 'guide me', 'help me', 'i\'d like to'
        ]
        
        # Check for positive responses
        is_positive = any(response in input_lower for response in positive_responses)
        
        # Also check for goal-type specific positive responses
        goal_specific_positive = [
            'lose weight', 'build muscle', 'gain muscle', 'maintain weight',
            'get stronger', 'get fit', 'start a goal'
        ]
        
        if is_positive or any(goal_phrase in input_lower for goal_phrase in goal_specific_positive):
            # Clear decline flags since user is now interested
            st.session_state.goal_suggestion_declined = False
            return True, self._generate_goal_setup_guidance()
        
        # Negative responses to goal setting (Requirement 4.4)
        negative_responses = [
            'no', 'nah', 'not now', 'maybe later', 'not interested', 'not ready',
            'don\'t want', 'no thanks', 'not yet', 'skip', 'pass', 'another time'
        ]
        
        if any(response in input_lower for response in negative_responses):
            # Track decline with enhanced information
            st.session_state.goal_suggestion_declined = True
            current_decline_count = st.session_state.get('goal_decline_count', 0)
            st.session_state.goal_decline_count = current_decline_count + 1
            st.session_state.last_decline_timestamp = {
                'conversation_count': conversation_count,
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate appropriate decline response based on decline count
            decline_responses = [
                "No problem! I'm here to help with general fitness advice anytime. Feel free to ask me about workouts, nutrition, or anything else! 😊",
                "That's totally fine! I'll focus on giving you great general fitness tips. You can always set a goal later if you change your mind! 💪",
                "Understood! I'm happy to help with any fitness questions you have. Setting goals is always optional - I'm here either way! 🌟"
            ]
            
            response_index = min(current_decline_count, len(decline_responses) - 1)
            return True, decline_responses[response_index]
        
        return False, ""
    
    def _generate_goal_setup_guidance(self) -> str:
        """
        Generate guidance for setting up a fitness goal.
        
        Returns:
            Goal setup guidance message
        """
        return (
            "Great! I'd love to help you set a fitness goal! 🎯\n\n"
            "To get started, I need to know:\n"
            "1. **Goal type**: Are you looking to lose weight, build muscle, or maintain your current weight?\n"
            "2. **Current stats**: What's your current weight?\n"
            "3. **Target**: What's your target weight or goal?\n"
            "4. **Timeline**: How many weeks do you want to work toward this goal? (4-52 weeks)\n\n"
            "You can set your goal using the sidebar, or just tell me your preferences and I'll guide you through it! 💪"
        )
    
    def detect_goal_decline_in_conversation(self, user_input: str) -> bool:
        """
        Detect if user is declining goal setting in their response.
        
        Args:
            user_input: User's input message
            
        Returns:
            True if goal decline detected
        """
        input_lower = user_input.lower()
        
        # Enhanced decline detection patterns
        decline_indicators = [
            'don\'t want to set', 'not interested in goals', 'no goals', 
            'just want advice', 'don\'t need a goal', 'skip the goal',
            'don\'t want goals', 'no goal setting', 'just general advice',
            'not ready for goals', 'don\'t like goals', 'goals aren\'t for me',
            'prefer no goals', 'just tips please', 'no tracking'
        ]
        
        # Check for explicit decline patterns
        explicit_decline = any(indicator in input_lower for indicator in decline_indicators)
        
        # Check for implicit decline patterns (when user asks for general advice after goal suggestion)
        implicit_decline_patterns = [
            'just tell me about', 'just want to know about', 'just give me',
            'only want', 'just need', 'can you just'
        ]
        
        # Only consider implicit decline if it follows a recent goal suggestion
        last_suggestion = st.session_state.get('last_goal_suggestion')
        conversation_count = st.session_state.get('conversation_count', 0)
        
        implicit_decline = False
        if last_suggestion and conversation_count - last_suggestion.get('conversation', 0) <= 1:
            implicit_decline = any(pattern in input_lower for pattern in implicit_decline_patterns)
        
        if explicit_decline or implicit_decline:
            # Track the decline
            conversation_count = st.session_state.get('conversation_count', 0)
            st.session_state.goal_suggestion_declined = True
            current_decline_count = st.session_state.get('goal_decline_count', 0)
            st.session_state.goal_decline_count = current_decline_count + 1
            st.session_state.last_decline_timestamp = {
                'conversation_count': conversation_count,
                'timestamp': datetime.now().isoformat()
            }
        
        return explicit_decline or implicit_decline
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about conversation patterns and goal suggestion effectiveness.
        
        Returns:
            Dictionary with conversation analytics
        """
        return {
            'conversation_count': st.session_state.get('conversation_count', 0),
            'goal_decline_count': st.session_state.get('goal_decline_count', 0),
            'goal_suggestion_declined': st.session_state.get('goal_suggestion_declined', False),
            'last_goal_suggestion': st.session_state.get('last_goal_suggestion'),
            'last_decline_timestamp': st.session_state.get('last_decline_timestamp'),
            'fitness_objectives_mentioned': len(st.session_state.get('fitness_objectives_mentioned', [])),
            'goal_suggestion_triggers': len(st.session_state.get('goal_suggestion_triggers', []))
        }
    
    def reset_goal_suggestion_state(self):
        """Reset goal suggestion state (useful for testing or user preference)."""
        st.session_state.goal_suggestion_declined = False
        st.session_state.goal_decline_count = 0
        st.session_state.last_goal_suggestion = None
        st.session_state.last_decline_timestamp = None
        st.session_state.fitness_objectives_mentioned = []
        st.session_state.goal_suggestion_triggers = []


def get_enhanced_chatbot_response(user_input: str, original_response: str, 
                                user_id: str, intent: str = None) -> str:
    """
    Main function to get enhanced chatbot response with goal awareness.
    
    This function integrates with the existing chatbot by taking the original
    response and enhancing it with goal-specific context and advice.
    
    Args:
        user_input: User's input message
        original_response: Response from original chatbot
        user_id: User identifier
        intent: Detected intent from original chatbot
        
    Returns:
        Enhanced response with goal awareness
    """
    try:
        # Initialize enhanced chatbot
        enhanced_chatbot = EnhancedChatbot()
        
        # Check if this is a response to goal setting suggestion
        handled, goal_response = enhanced_chatbot.handle_goal_response(user_input, user_id)
        if handled:
            return goal_response
        
        # Check if user is declining goal setting in conversation
        if enhanced_chatbot.detect_goal_decline_in_conversation(user_input):
            st.session_state.goal_suggestion_declined = True
            current_decline_count = st.session_state.get('goal_decline_count', 0)
            st.session_state.goal_decline_count = current_decline_count + 1
        
        # Get enhanced response with goal context
        return enhanced_chatbot.get_enhanced_response(
            user_input, original_response, user_id, intent
        )
        
    except Exception as e:
        logger.error(f"Error in get_enhanced_chatbot_response: {str(e)}")
        return original_response


def should_suggest_goal_setting(conversation_history: List[Dict[str, str]]) -> bool:
    """
    Determine if goal setting should be suggested based on conversation history.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        True if goal setting should be suggested
    """
    try:
        # Check conversation count
        conversation_count = len(conversation_history)
        
        # Don't suggest if user has declined recently
        if st.session_state.get('goal_suggestion_declined', False):
            decline_count = st.session_state.get('goal_decline_count', 0)
            min_conversations = 10 + (decline_count * 5)
            if conversation_count < min_conversations:
                return False
        
        # Suggest within first 3 interactions
        if conversation_count <= 3:
            return True
        
        # Suggest periodically
        if conversation_count % 15 == 0:
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error in should_suggest_goal_setting: {str(e)}")
        return False


def generate_goal_specific_advice(intent: str, goal: FitnessGoal) -> str:
    """
    Generate goal-specific advice for a given intent and goal.
    
    Args:
        intent: Detected intent from chatbot
        goal: User's fitness goal
        
    Returns:
        Goal-specific advice string
    """
    try:
        recommendation_engine = RecommendationEngine()
        
        # Map intents to recommendation types
        if intent in ['workout_plan', 'beginner_workout', 'strength_workout', 'cardio_workout']:
            workout_type = intent.replace('_workout', '').replace('workout_plan', 'general')
            return recommendation_engine.get_workout_recommendations(goal, workout_type)
        
        elif intent in ['nutrition_plan', 'breakfast_ideas', 'lunch_ideas', 'dinner_ideas', 'snack_ideas']:
            meal_type = intent.replace('_ideas', '').replace('_plan', '')
            if meal_type == 'nutrition':
                meal_type = 'general'
            return recommendation_engine.get_nutrition_recommendations(goal, meal_type)
        
        elif intent == 'motivation':
            neutral_progress = {'status': 'on_track', 'progress_percentage': 50}
            return recommendation_engine.get_motivational_message(goal, neutral_progress)
        
        else:
            return f"For your {goal.goal_type.replace('_', ' ')} goal, focus on consistency and patience. Every step counts!"
        
    except Exception as e:
        logger.error(f"Error generating goal-specific advice: {str(e)}")
        return ""


def check_goal_milestones(goal: FitnessGoal, progress_history: List[Any]) -> Optional[str]:
    """
    Check for goal milestones and return celebration message.
    
    Args:
        goal: User's fitness goal
        progress_history: List of progress entries (not used in basic implementation)
        
    Returns:
        Milestone message or None
    """
    try:
        # Calculate time-based progress
        total_days = (goal.target_date - goal.created_date).days
        days_elapsed = total_days - goal.days_remaining()
        time_progress = (days_elapsed / total_days) if total_days > 0 else 0
        
        # Check for major milestones
        if 0.24 <= time_progress < 0.26:
            return "🎉 You're 25% through your fitness journey! Keep up the momentum!"
        elif 0.49 <= time_progress < 0.51:
            return "🎉 Halfway there! You're 50% through your goal timeline!"
        elif 0.74 <= time_progress < 0.76:
            return "🎉 You're in the final stretch! 75% complete!"
        elif time_progress >= 0.95 and not goal.is_overdue():
            return "🎉 Almost there! You're so close to reaching your goal!"
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking goal milestones: {str(e)}")
        return None