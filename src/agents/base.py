"""Base agent class for all HydroAssist agents."""

from abc import ABC, abstractmethod
from src.core.state import AgentState
from src.core.config import AppConfig
from src.utils.logger import setup_logger


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AppConfig):
        """
        Initialize base agent.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """
        Execute agent logic and return updated state.

        Args:
            state: Current conversation state

        Returns:
            Updated state with new messages and metadata
        """
        pass

    def _log_execution(self, state: AgentState, action: str, details: str = ""):
        """
        Log agent execution for debugging.

        Args:
            state: Current state
            action: Action being performed
            details: Additional details
        """
        intent = state.get('user_intent', 'unknown')
        aquifer = state.get('aquifer_context', 'unknown')

        log_msg = (
            f"Agent: {self.__class__.__name__} | "
            f"Action: {action} | "
            f"Intent: {intent} | "
            f"Aquifer: {aquifer}"
        )

        if details:
            log_msg += f" | {details}"

        self.logger.info(log_msg)
