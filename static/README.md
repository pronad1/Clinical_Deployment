# Static Assets

This directory contains static files for the web interface including stylesheets, JavaScript, images, and other assets.

## Directory Structure

```
static/
├── css/
│   └── main.css          # Main stylesheet with modern design system
├── js/
│   └── main.js           # Application JavaScript
├── images/               # Image assets (icons, logos, etc.)
└── fonts/                # Custom fonts (if any)
```

## Design System

### Professional Medical Theme
The interface uses a modern, clean medical/clinical design with:
- Professional medical blue color palette
- Clean typography with excellent readability
- Smooth animations and transitions
- Responsive grid layouts
- Accessible color contrasts (WCAG AA compliant)

### Color Palette
- **Primary Blue**: `#0066cc` - Main brand color
- **Primary Dark**: `#004d99` - Darker variant
- **Accent Teal**: `#00a896` - Complementary accent
- **Success Green**: `#10b981` - Positive states
- **Danger Red**: `#ef4444` - Alerts and abnormal results
- **Neutral Grays**: Multiple shades for text and backgrounds

### Typography
- **Font Family**: System font stack (-apple-system, BlinkMacSystemFont, Segoe UI, Roboto)
- **Base Size**: 16px (1rem)
- **Scale**: Modular scale for consistent sizing
- **Line Height**: 1.6 for optimal readability

### Components
- **Buttons**: Multiple variants (primary, secondary, success, ghost)
- **Cards**: Shadow-based elevation with hover effects
- **Metrics**: Clean boxes with hover interactions
- **Badges**: Status indicators for results
- **Alerts**: Color-coded messaging system

### Responsive Design
- **Mobile First**: Optimized for all screen sizes
- **Breakpoints**:
  - Mobile: < 768px
  - Tablet: 768px - 1024px
  - Desktop: > 1024px

## File Organization

### CSS Architecture
The main.css file follows a logical structure:
1. **CSS Variables**: Design tokens and theme variables
2. **Reset & Base**: Normalize and base styles
3. **Layout**: Container and grid systems
4. **Components**: Reusable UI components
5. **Utilities**: Helper classes
6. **Responsive**: Media queries

### JavaScript Modules
The main.js file is organized into:
1. **Initialization**: Setup and event bindings
2. **File Handling**: Upload and drag-drop logic
3. **API Communication**: Fetch requests to backend
4. **UI Updates**: Dynamic content rendering
5. **Visualizations**: XAI generation functions
6. **Downloads**: File download utilities

## Best Practices

### CSS
- Use CSS variables for consistency
- Follow BEM-like naming conventions
- Mobile-first responsive design
- Optimize for performance (minimal repaints)

### JavaScript
- Vanilla JS (no framework dependencies)
- Async/await for cleaner async code
- Error handling for all network requests
- Separated concerns (DOM, logic, API)

### Performance
- **CSS**: Single concatenated file, minified for production
- **JavaScript**: Modular structure, can be split for lazy loading
- **Images**: Served as base64 from backend to reduce requests
- **Caching**: Static assets should be cached with appropriate headers

## Development

### Local Development
Files are served from this directory by Flask's static file handler.

### Production
For production deployment:
1. Minify CSS and JavaScript
2. Enable gzip compression
3. Set appropriate cache headers
4. Consider CDN for static assets

## Browser Support
- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile: iOS Safari, Chrome Mobile

## Accessibility
- ARIA labels on interactive elements
- Keyboard navigation support
- Screen reader friendly
- High contrast mode compatible
- Focus indicators on all interactive elements

---
*Last Updated: February 2026*
