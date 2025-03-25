// Theme icons
const sunIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
</svg>`;

const moonIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
</svg>`;

// Theme management
class ThemeManager {
  constructor() {
    this.theme = localStorage.getItem('theme') || 'light';
    this.prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    this.init();
  }

  init() {
    // Create theme switch button
    this.createThemeSwitch();
    
    // Set initial theme
    this.setTheme(this.theme);
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
      this.prefersDark = e.matches;
      if (!localStorage.getItem('theme')) {
        this.setTheme(this.prefersDark ? 'dark' : 'light');
      }
    });
  }

  createThemeSwitch() {
    const button = document.createElement('button');
    button.className = 'theme-switch';
    button.setAttribute('aria-label', 'Toggle theme');
    button.innerHTML = this.theme === 'dark' ? sunIcon : moonIcon;
    button.addEventListener('click', () => this.toggleTheme());
    document.body.appendChild(button);
  }

  setTheme(theme) {
    this.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update button icon
    const button = document.querySelector('.theme-switch');
    if (button) {
      button.innerHTML = theme === 'dark' ? sunIcon : moonIcon;
    }
  }

  toggleTheme() {
    const newTheme = this.theme === 'dark' ? 'light' : 'dark';
    this.setTheme(newTheme);
  }
}

// Initialize theme manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new ThemeManager();
}); 