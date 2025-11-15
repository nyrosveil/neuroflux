import React, { useEffect, useRef } from 'react';
import { useDrag } from '@use-gesture/react';
import './BottomSheet.css';

function BottomSheet({ isOpen, onClose, children, title, height = '60vh' }) {
  const sheetRef = useRef(null);
  const overlayRef = useRef(null);

  // Drag gesture for bottom sheet
  const bind = useDrag(
    ({ down, movement: [, my], velocity }) => {
      if (!sheetRef.current) return;

      const sheet = sheetRef.current;
      const overlay = overlayRef.current;

      if (down) {
        // During drag
        const newY = Math.max(0, my);
        sheet.style.transform = `translateY(${newY}px)`;
        overlay.style.opacity = Math.max(0.1, 1 - (newY / 300));
      } else {
        // After drag release
        const shouldClose = my > 100 || (my > 50 && velocity > 0.5);

        if (shouldClose) {
          closeSheet();
        } else {
          // Snap back to open position
          sheet.style.transform = 'translateY(0)';
          overlay.style.opacity = '1';
        }
      }
    },
    {
      axis: 'y',
      filterTaps: true,
      bounds: { top: 0 },
    }
  );

  const closeSheet = () => {
    if (!sheetRef.current || !overlayRef.current) return;

    const sheet = sheetRef.current;
    const overlay = overlayRef.current;

    sheet.style.transform = 'translateY(100%)';
    overlay.style.opacity = '0';

    setTimeout(() => {
      onClose();
    }, 300);
  };

  useEffect(() => {
    if (isOpen && sheetRef.current && overlayRef.current) {
      // Animate in
      const sheet = sheetRef.current;
      const overlay = overlayRef.current;

      // Force reflow
      void sheet.offsetHeight;

      sheet.style.transform = 'translateY(0)';
      overlay.style.opacity = '1';
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="bottom-sheet-overlay" ref={overlayRef} onClick={closeSheet}>
      <div
        className="bottom-sheet"
        ref={sheetRef}
        style={{ height }}
        onClick={(e) => e.stopPropagation()}
        {...bind()}
      >
        <div className="bottom-sheet-header">
          <div className="drag-handle" />
          {title && <h3 className="sheet-title">{title}</h3>}
          <button className="sheet-close-btn" onClick={closeSheet}>
            ✕
          </button>
        </div>

        <div className="bottom-sheet-content">
          {children}
        </div>
      </div>
    </div>
  );
}

export default BottomSheet;