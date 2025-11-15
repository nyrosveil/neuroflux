import React, { useState, useEffect } from 'react';
import './OnboardingTour.css';

function OnboardingTour({ isVisible, onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);

  const steps = [
    {
      title: "Welcome to NeuroFlux",
      content: "Your AI-powered trading dashboard is ready. Let's take a quick tour to get you started.",
      target: null,
      position: "center"
    },
    {
      title: "Agent Control Panel",
      content: "Here you can start, stop, and configure your trading agents. Each agent has specific capabilities.",
      target: ".agent-control-panel",
      position: "bottom"
    },
    {
      title: "System Metrics",
      content: "Monitor your system's health, agent performance, and overall trading statistics.",
      target: ".agent-metrics",
      position: "bottom"
    },
    {
      title: "Market Data",
      content: "View real-time market data, order books, and set price alerts for your favorite tokens.",
      target: ".market-data-widget",
      position: "left"
    },
    {
      title: "AI Predictions",
      content: "See AI-powered price predictions with confidence intervals and market sentiment analysis.",
      target: ".prediction-charts",
      position: "left"
    },
    {
      title: "Mobile Navigation",
      content: "Swipe between views or use the bottom navigation tabs on mobile devices.",
      target: ".mobile-nav",
      position: "top"
    },
    {
      title: "You're All Set!",
      content: "NeuroFlux is now ready for trading. Remember to monitor your agents and set appropriate risk limits.",
      target: null,
      position: "center"
    }
  ];

  useEffect(() => {
    if (isVisible && !isCompleted) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }

    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [isVisible, isCompleted]);

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      completeTour();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const completeTour = () => {
    setIsCompleted(true);
    localStorage.setItem('neuroflux_onboarding_completed', 'true');
    onComplete();
  };

  const skipTour = () => {
    completeTour();
  };

  if (!isVisible || isCompleted) return null;

  const currentStepData = steps[currentStep];
  const targetElement = currentStepData.target ?
    document.querySelector(currentStepData.target) : null;

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-backdrop" onClick={skipTour} />

      <div
        className={`onboarding-tooltip ${currentStepData.position}`}
        style={targetElement ? getTooltipPosition(targetElement, currentStepData.position) : {}}
      >
        <div className="tooltip-header">
          <h3>{currentStepData.title}</h3>
          <button className="tooltip-close" onClick={skipTour}>×</button>
        </div>

        <div className="tooltip-content">
          <p>{currentStepData.content}</p>
        </div>

        <div className="tooltip-footer">
          <div className="tooltip-progress">
            <div className="progress-dots">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={`progress-dot ${index === currentStep ? 'active' : ''} ${index < currentStep ? 'completed' : ''}`}
                />
              ))}
            </div>
            <span className="progress-text">
              {currentStep + 1} of {steps.length}
            </span>
          </div>

          <div className="tooltip-actions">
            {currentStep > 0 && (
              <button className="tooltip-btn secondary" onClick={prevStep}>
                Previous
              </button>
            )}
            <button className="tooltip-btn primary" onClick={nextStep}>
              {currentStep === steps.length - 1 ? 'Get Started' : 'Next'}
            </button>
          </div>
        </div>

        {/* Arrow pointer */}
        {targetElement && (
          <div className={`tooltip-arrow ${currentStepData.position}`} />
        )}
      </div>
    </div>
  );
}

function getTooltipPosition(targetElement, position) {
  if (!targetElement) return {};

  const rect = targetElement.getBoundingClientRect();
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

  switch (position) {
    case 'top':
      return {
        top: rect.top + scrollTop - 10,
        left: rect.left + scrollLeft + (rect.width / 2),
        transform: 'translateX(-50%) translateY(-100%)'
      };
    case 'bottom':
      return {
        top: rect.bottom + scrollTop + 10,
        left: rect.left + scrollLeft + (rect.width / 2),
        transform: 'translateX(-50%)'
      };
    case 'left':
      return {
        top: rect.top + scrollTop + (rect.height / 2),
        left: rect.left + scrollLeft - 10,
        transform: 'translateX(-100%) translateY(-50%)'
      };
    case 'right':
      return {
        top: rect.top + scrollTop + (rect.height / 2),
        left: rect.right + scrollLeft + 10,
        transform: 'translateY(-50%)'
      };
    default:
      return {};
  }
}

// Hook to manage onboarding state
export const useOnboarding = () => {
  const [showTour, setShowTour] = useState(false);

  useEffect(() => {
    const hasCompletedTour = localStorage.getItem('neuroflux_onboarding_completed');
    if (!hasCompletedTour) {
      // Delay showing tour to let components load
      const timer = setTimeout(() => {
        setShowTour(true);
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, []);

  const completeTour = () => {
    setShowTour(false);
  };

  const restartTour = () => {
    localStorage.removeItem('neuroflux_onboarding_completed');
    setShowTour(true);
  };

  return {
    showTour,
    completeTour,
    restartTour
  };
};

export default OnboardingTour;