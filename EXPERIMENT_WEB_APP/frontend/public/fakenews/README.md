# Fake News Experiment Assets

Place your media assets in the following folders:

## Folder Structure

```
fakenews/
├── logos/
│   └── logo.png          # Source logo (used for all items currently)
└── thumbnails/
    ├── image.jpg         # Test image (used for odd items: 1,3,5,7,9,11,13,15)
    └── video.mp4         # Test video (used for even items: 2,4,6,8,10,12,14,16)
```

## Required Files

### logos/logo.png
- Recommended size: 100x100px
- Format: PNG with transparency
- Used as the source logo for all news items

### thumbnails/image.jpg
- Recommended size: 800x400px (2:1 aspect ratio)
- Format: JPG
- Used for items: fake_01, fake_03, fake_05, fake_07, real_01, real_03, real_05, real_07

### thumbnails/video.mp4
- Recommended size: 800x400px
- Format: MP4 (H.264)
- Keep file size small (< 5MB) for quick loading
- Video autoplays muted and loops
- Used for items: fake_02, fake_04, fake_06, fake_08, real_02, real_04, real_06, real_08

## Notes

- Videos autoplay muted with loop enabled
- If an asset fails to load, a placeholder will be shown
- Update the seed script (`backend/scripts/seed_fake_news.py`) to use specific filenames when ready
