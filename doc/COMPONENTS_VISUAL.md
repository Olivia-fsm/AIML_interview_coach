# Visual & Theming Components

## `VisualLab.tsx`
A creative studio for generating study aids.
- **Modes**:
    - **Create**: Text-to-Image (Diagram generation).
    - **Edit**: Image-to-Image (Modifying diagrams/notes).
    - **Video**: Text/Image-to-Video (Veo model animation).
- **UI**: Split pane (Input controls vs. Result preview). Handles file uploads for edit/video modes.

## Theming System
Controlled by `App.tsx` and CSS variables.
- **`ThemeSelector.tsx`**: Grid of available themes (Midnight, Solar, Neon, Deepspace, Toon, Cosmic, Sea, Flower, Snow, Gothic).
- **`ClickEffects.tsx`**: Global particle system. Renders different particles based on the active theme (e.g., Stars for Cosmic, Bubbles for Sea, Lightning for Gothic).

## Canvas Backgrounds
Standalone animated backgrounds rendered behind the UI.
- **`CosmicBackground.tsx`**: Parallax stars and orbiting planets.
- **`SeaBackground.tsx`**: Underwater gradient, bubbles, light rays, and floating character.
- **`FlowerBackground.tsx`**: Swaying flowers, falling petals, and a sleeping star character.
- **`SnowBackground.tsx`**: Falling snow with mouse wind interaction and a snowman.
- **`GothicBackground.tsx`**: Dark atmosphere, fog, silhouette castle, and floating feathers.

## `WelcomePage.tsx`
The landing experience.
- **Design**: Glassmorphism cards, animated gradients.
- **Function**: Introduces features and routes user to theme selection.
