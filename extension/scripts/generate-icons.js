// This script is a helper for generating icons in various sizes required by Chrome extensions.
// You can run it with: node scripts/generate-icons.js
// Requires sharp package: npm install sharp

import fs from "fs";
import path from "path";
import sharp from "sharp";

const SOURCE_LOGO = path.join(__dirname, "../public/Neoterik-Genesis.png");
const ICONS_DIR = path.join(__dirname, "../icons");

// Create icons directory if it doesn't exist
if (!fs.existsSync(ICONS_DIR)) {
  fs.mkdirSync(ICONS_DIR, { recursive: true });
}

// Icon sizes required for Chrome extensions
const ICON_SIZES = [16, 32, 48, 128];

async function generateIcons() {
  try {
    console.log("Generating extension icons...");

    for (const size of ICON_SIZES) {
      const outputPath = path.join(ICONS_DIR, `icon${size}.png`);

      await sharp(SOURCE_LOGO).resize(size, size).toFile(outputPath);

      console.log(`Created icon${size}.png`);
    }

    console.log("All icons generated successfully!");
  } catch (error) {
    console.error("Error generating icons:", error);
    process.exit(1);
  }
}

generateIcons();
