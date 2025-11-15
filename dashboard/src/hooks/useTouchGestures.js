import { useDrag, usePinch } from '@use-gesture/react';
import { useCallback } from 'react';

export const useTouchGestures = (config = {}) => {
  const {
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onPinch,
    onTap,
    onDoubleTap,
    onLongPress,
    swipeThreshold = 50,
    longPressDelay = 500,
  } = config;

  // Swipe gesture handler
  const bindSwipe = useDrag(
    ({ down, movement: [mx, my], direction: [xDir, yDir], velocity }) => {
      if (!down && Math.abs(mx) > swipeThreshold) {
        // Horizontal swipe
        if (xDir > 0 && Math.abs(mx) > Math.abs(my)) {
          onSwipeRight?.();
        } else if (xDir < 0 && Math.abs(mx) > Math.abs(my)) {
          onSwipeLeft?.();
        }
        // Vertical swipe
        else if (yDir > 0 && Math.abs(my) > Math.abs(mx)) {
          onSwipeDown?.();
        } else if (yDir < 0 && Math.abs(my) > Math.abs(mx)) {
          onSwipeUp?.();
        }
      }
    },
    {
      axis: undefined, // Allow both axes
      filterTaps: true,
      threshold: swipeThreshold,
    }
  );

  // Pinch gesture handler for zoom
  const bindPinch = usePinch(
    ({ offset: [scale] }) => {
      onPinch?.(scale);
    },
    {
      scaleBounds: { min: 0.5, max: 3 },
    }
  );

  // Tap gesture handler - disabled due to library version issue
  const bindTap = () => ({});

  // Long press gesture handler - disabled due to library version issue
  const bindLongPress = () => ({});


  return {
    bindSwipe,
    bindPinch,
    bindTap,
    bindLongPress,
    // Combined binding for components that need multiple gestures
    bindAll: useCallback(
      () => ({
        ...bindSwipe(),
        ...bindPinch(),
        ...bindTap(),
        ...bindLongPress(),
      }),
      [bindSwipe, bindPinch, bindTap, bindLongPress]
    ),
  };
};

// Touch feedback utility
export const useTouchFeedback = () => {
  const addTouchFeedback = useCallback((element) => {
    if (!element) return;

    const handleTouchStart = () => {
      element.style.transform = 'scale(0.95)';
      element.style.transition = 'transform 0.1s ease';
    };

    const handleTouchEnd = () => {
      element.style.transform = 'scale(1)';
    };

    element.addEventListener('touchstart', handleTouchStart, { passive: true });
    element.addEventListener('touchend', handleTouchEnd, { passive: true });
    element.addEventListener('touchcancel', handleTouchEnd, { passive: true });

    // Cleanup function
    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchend', handleTouchEnd);
      element.removeEventListener('touchcancel', handleTouchEnd);
    };
  }, []);

  return { addTouchFeedback };
};

// Pull-to-refresh hook
export const usePullToRefresh = (onRefresh, threshold = 80) => {
  const bind = useDrag(
    ({ down, movement: [, my], cancel }) => {
      if (my > threshold && !down) {
        onRefresh?.();
      }
    },
    {
      axis: 'y',
      filterTaps: true,
      threshold: threshold,
    }
  );

  return bind;
};