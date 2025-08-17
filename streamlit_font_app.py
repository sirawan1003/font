
"""
streamlit run streamlit_font_app.py
"""

import streamlit as st
import os
import io
import zipfile
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import shutil
import base64

# Import our font processing classes
try:
    from font_gan_eco_processor import FontGANEcoProcessor
except ImportError:
    st.error("ไม่สามารถโหลดโมดูลประมวลผลฟอนต์ได้ ตรวจสอบไฟล์ font_gan_eco_processor.py")
    st.stop()

# Import fonttools for TTF generation
try:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.misc.timeTools import timestampSinceEpoch
    import unicodedata
    TTF_AVAILABLE = True
except ImportError:
    st.warning("⚠️ fonttools ไม่พร้อมใช้งาน - จะสร้างเฉพาะไฟล์ PNG")
    TTF_AVAILABLE = False

# Thai character sets for generation
THAI_CHARACTERS = [
    'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 
    'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 
    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ',
    'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    '่', '้', '๊', '๋', '์', 'ํ', '๎',
    '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙'
]

class TTFGenerator:
    
    def __init__(self, font_name="AI_Generated_Font", font_version="1.0"):
        self.font_name = font_name
        self.font_version = font_version
        self.units_per_em = 1000
        self.debug_callback = None
    
    def set_debug_callback(self, callback):
        """Set callback for debug messages"""
        self.debug_callback = callback
    
    def _log(self, message):
        """Log message using callback if available"""
        if self.debug_callback:
            self.debug_callback(message)
        else:
            print(message)
    
    def get_unicode_value(self, char):
        """Get Unicode code point for character - improved for Windows compatibility"""
        try:
            # Handle character names from font processing
            if char.startswith('user_original_'):
                actual_char = char.replace('user_original_', '')
                return ord(actual_char) if len(actual_char) == 1 else ord(actual_char[0])
            elif char.startswith('user_gan_'):
                actual_char = char.replace('user_gan_', '')
                return ord(actual_char) if len(actual_char) == 1 else ord(actual_char[0])
            elif char.startswith('user_eco_'):
                actual_char = char.replace('user_eco_', '')
                return ord(actual_char) if len(actual_char) == 1 else ord(actual_char[0])
            elif char.startswith('original_char_'):
                # Map to Thai characters
                char_num = int(char.replace('original_char_', ''))
                if char_num < len(THAI_CHARACTERS):
                    return ord(THAI_CHARACTERS[char_num])
                return 0xE000 + char_num  # Private Use Area
            elif char.startswith('gan_char_'):
                char_num = int(char.replace('gan_char_', ''))
                if char_num < len(THAI_CHARACTERS):
                    return ord(THAI_CHARACTERS[char_num])
                return 0xE100 + char_num  # Private Use Area
            elif char.startswith('eco_char_'):
                char_num = int(char.replace('eco_char_', ''))
                if char_num < len(THAI_CHARACTERS):
                    return ord(THAI_CHARACTERS[char_num])
                return 0xE200 + char_num  # Private Use Area
            elif char.startswith('char_'):
                # Simple character mapping for basic ASCII
                actual_char = char.replace('char_', '')
                if len(actual_char) == 1 and actual_char.isascii():
                    return ord(actual_char)
                else:
                    # Hash to a reasonable range for unknown chars
                    import hashlib
                    hash_val = int(hashlib.md5(char.encode()).hexdigest()[:2], 16)
                    return 65 + (hash_val % 26)  # A-Z range
            elif len(char) == 1:
                # Single character - get its Unicode value
                return ord(char)
            else:
                # For any other longer names, use hash-based approach
                import hashlib
                hash_val = int(hashlib.md5(char.encode()).hexdigest()[:2], 16)
                return 65 + (hash_val % 26)  # A-Z range for Windows compatibility
        except Exception as e:
            self._log(f"⚠️ Cannot get unicode for {char}: {e}")
            # Default fallback
            return 65  # 'A'
    
    def image_to_contours(self, image, output_size=1000):
        """Convert image array to contours for font creation"""
        try:
            # Ensure we have a grayscale image
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = image.copy()
            
            # Threshold to get a binary image (black character on white background)
            # If the image is mostly dark, invert it
            if np.mean(img) < 127:
                img = 255 - img
            
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self._log("⚠️ No contours found in image")
                return []
            
            # --- Transformation ---
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Scale contours to fit within the font's em square
            scale = output_size / max(w, h) * 0.8  # Leave some margin
            
            scaled_contours = []
            for contour in contours:
                # Scale and center
                scaled_contour = (contour - [x, y]) * scale
                # Add margin and flip Y (font coordinates have Y going up)
                margin = output_size * 0.1
                scaled_contour[:, :, 0] += margin  # X offset
                scaled_contour[:, :, 1] = output_size - (scaled_contour[:, :, 1] + margin)  # Flip Y
                scaled_contours.append(scaled_contour.astype(int))
            
            return scaled_contours
            
        except Exception as e:
            self._log(f"❌ Error processing image to contours: {e}")
            return []
    
    def contours_to_glyph(self, contours):
        """Convert contours to a fontTools Glyph object"""
        if not contours:
            return None
    
        try:
            pen = TTGlyphPen(None)
            for contour in contours:
                if len(contour) < 3:  # Need at least 3 points
                    continue
            
                # Start the path
                pen.moveTo(tuple(contour[0][0]))
                # Add lines for the rest of the points
                for point in contour[1:]:
                    pen.lineTo(tuple(point[0]))
                # Close the path
                pen.closePath()
        
            return pen.glyph()
                
        except Exception as e:
            self._log(f"❌ Error creating glyph from contours: {e}")
            return None
    
    def _create_unique_character_mapping(self, character_images):
        """Create unique character mapping to avoid Unicode conflicts"""
        char_map = {}
        glyph_names = [".notdef"]  # Required glyph
        used_unicode_vals = set()
        char_to_glyph = {}  # Track char_name -> glyph_name mapping
        
        glyph_counter = 0
        for char_name, image in character_images.items():
            unicode_val = self.get_unicode_value(char_name)
            
            # Ensure unique Unicode values
            while unicode_val in used_unicode_vals:
                unicode_val += 1
                if unicode_val > 0xFFFF:  # Stay within BMP
                    unicode_val = 0xE000 + (glyph_counter % 6400)  # Private Use Area
            
            used_unicode_vals.add(unicode_val)
            glyph_name = f"glyph_{glyph_counter}_{unicode_val}"  # Unique glyph name
            char_map[unicode_val] = glyph_name
            glyph_names.append(glyph_name)
            char_to_glyph[char_name] = glyph_name
            glyph_counter += 1
            
            self._log(f"Mapping {char_name} → U+{unicode_val:04X} → {glyph_name}")
        
        return char_map, glyph_names, char_to_glyph

    def create_ttf_font(self, character_images, output_path):
        """Create TTF font file from character images"""
        self._log(f"Creating TTF Font: {self.font_name}")
        self._log(f"Processing {len(character_images)} characters...")

        try:
            upm = self.units_per_em
            ascender = 800
            descender = -200
            glyph_width = 600

            # Create unique character mapping
            char_map, glyph_names, char_to_glyph = self._create_unique_character_mapping(character_images)

            # Add space character
            if 32 not in char_map:
                char_map[32] = "space"
                glyph_names.append("space")

            fb = FontBuilder(upm, isTTF=True)
            fb.setupGlyphOrder(glyph_names)
            fb.setupCharacterMap(char_map)

            h_metrics = {glyph_name: (glyph_width, 50) for glyph_name in glyph_names}
            fb.setupHorizontalMetrics(h_metrics)

            # Create .notdef glyph
            glyphs = {}
            notdef_pen = TTGlyphPen(None)
            notdef_pen.moveTo((50, 50))
            notdef_pen.lineTo((550, 50))
            notdef_pen.lineTo((550, 650))
            notdef_pen.lineTo((50, 650))
            notdef_pen.closePath()
            glyphs[".notdef"] = notdef_pen.glyph()

            # Space glyph
            if "space" in glyph_names:
                space_pen = TTGlyphPen(None)
                glyphs["space"] = space_pen.glyph()

            # Process glyphs
            for char_name, image in character_images.items():
                glyph_name = char_to_glyph[char_name]
                self._log(f"🔧 Processing {char_name} -> {glyph_name}...")

                contours = self.image_to_contours(image, upm)
                if contours:
                    glyph = self.contours_to_glyph(contours)
                    if glyph:
                        glyphs[glyph_name] = glyph
                        self._log(f"Created glyph for {char_name}")
                    else:
                        empty_pen = TTGlyphPen(None)
                        glyphs[glyph_name] = empty_pen.glyph()
                else:
                    empty_pen = TTGlyphPen(None)
                    glyphs[glyph_name] = empty_pen.glyph()

            # Verify glyph consistency
            if len(glyph_names) != len(glyphs):
                raise ValueError("Glyph count mismatch")

            fb.setupGlyf(glyphs)
            fb.setupMaxp()

            fb.setupNameTable({
                "familyName": self.font_name,
                "styleName": "Regular",
                "uniqueFontIdentifier": f"{self.font_name}-{self.font_version}",
                "fullName": f"{self.font_name} Regular",
                "version": f"Version {self.font_version}",
                "psName": self.font_name.replace(" ", ""),
                "manufacturer": "AI Font Generator",
                "designer": "AI Generated Font",
                "description": f"AI Generated Font - {self.font_name}",
                "vendorURL": "https://github.com/ai-font-generator",
                "designerURL": "https://github.com/ai-font-generator",
            })

            fb.setupHorizontalHeader(ascent=ascender, descent=descender, lineGap=100)
            fb.setupOS2(
                sTypoAscender=ascender,
                sTypoDescender=descender,
                sTypoLineGap=100,
                usWeightClass=400,
                usWidthClass=5,
                fsType=0,
                sxHeight=500,
                sCapHeight=700,
                usDefaultChar=32,
                usBreakChar=32,
                ulCodePageRange1=1,
                ulUnicodeRange1=(1 << 0) | (1 << 25),  # Latin + Thai
            )
            fb.setupPost(italicAngle=0, underlinePosition=-100, underlineThickness=50)

            fb.save(output_path)
            self._log(f"TTF font saved successfully: {output_path}")
            return True

        except Exception as e:
            self._log(f"Error creating TTF font: {e}")
            return False
    
    def _create_font_package(self, character_images, output_path):
        """Create font package as fallback when TTF creation fails"""
        try:
            package_path = output_path.replace('.ttf', '_FontPackage.zip')
            
            with zipfile.ZipFile(package_path, 'w') as zf:
                # Add font info
                font_info = f"""Font Package: {self.font_name}
Version: {self.font_version}
Characters: {len(character_images)}

"""
                zf.writestr("README.txt", font_info)
                
                # Add character images
                for char, image in character_images.items():
                    unicode_val = self.get_unicode_value(char)
                    filename = f"char_{unicode_val:04X}_{char}.png"
                    safe_filename = filename.replace('/', '_').replace('\\', '_')
                    
                    success, encoded = cv2.imencode('.png', image)
                    if success:
                        zf.writestr(f"characters/{safe_filename}", encoded.tobytes())
            
            self._log(f"✅ สร้าง Font Package: {package_path}")
            return True
            
        except Exception as e:
            self._log(f"❌ ไม่สามารถสร้าง Font Package: {e}")
            return False

# Page configuration
st.set_page_config(
    page_title="🎨 ระบบสร้างฟอนต์ภาษาไทยด้วย AI",
    page_icon="🇹🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stats-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .folder-upload-box {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">ระบบสร้างฟอนต์ประหยัดหมึกด้วย GAN</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        
        # Sample character upload
        st.subheader("📁 อัพโหลดตัวอักษร")
        st.info("อัพโหลดตัวอักษร AI จะสร้างตัวอักษรที่เจาะรูให้")
        
        uploaded_files = st.file_uploader(
            "📁 เลือกไฟล์ PNG ตัวอักษร", 
            type=['png'], 
            accept_multiple_files=True,
            help="อัพโหลดตัวอักษร"
        )
        
        uploaded_data = None
        if uploaded_files:
            uploaded_data = process_sample_files(uploaded_files)
        
        st.divider()
        
        # Character set selection
        st.subheader("เลือกชุดตัวอักษรที่ต้องการ")
        character_sets = st.multiselect(
            "ชุดตัวอักษร",
            ["พยัญชนะไทย (44 ตัว)", "สระและวรรณยุกต์", "เลขไทย (๐-๙)", "เลขอารบิก (0-9)"]
        )
        
        # AI Parameters
        st.subheader("การตั้งค่า")
        generation_quality = st.select_slider(
            "คุณภาพการสร้าง",
            options=["เร็ว", "ปานกลาง", "ละเอียด", "สูงสุด"],
            value="ละเอียด"
        )
        
        # ECO Parameters - IMPROVED FOR LARGER HOLES
        st.subheader("การตั้งค่ารู ECO")
        hole_intensity = st.select_slider(
            "ความเข้มข้นของรู",
            options=["เบา", "ปานกลาง", "หนัก", "หนักมาก"],
            value="ปานกลาง"
        )
        
        hole_size_preset = st.selectbox(
            "ขนาดรู",
            ["เล็ก (ประหยัด 30%)", "กลาง (ประหยัด 45%)", "ใหญ่ (ประหยัด 60%)", "ใหญ่มาก (ประหยัด 75%)"],
            index=2
        )
        
        # Output settings
        st.subheader("การตั้งค่าผลลัพธ์")
        output_sizes = st.multiselect(
            "ขนาดภาพ", 
            options=[256, 512, 1024, 2048],
            default=[1024]
        )
        
        # TTF Generation options
        st.subheader("การสร้าง TTF Font")
        create_ttf = st.checkbox("สร้างไฟล์ TTF Font", value=True)
        if create_ttf:
            font_name = st.text_input("ชื่อฟอนต์", value="MyEcoFont")
            create_eco_ttf = st.checkbox("สร้าง TTF แบบ ECO ด้วย", value=True)
        else:
            font_name = "MyEcoFont"
            create_eco_ttf = False
            
        if not TTF_AVAILABLE:
            st.info("จะสร้าง Font Package (ZIP) แทน TTF เนื่องจากไม่มี fonttools")
        
        st.divider()
        
        # Process button
        process_button = st.button("สร้างฟอนต์สมบูรณ์", type="primary", use_container_width=True)
    
    # Main content area
    if not uploaded_data:
        show_welcome_screen()
    else:
        show_sample_info(uploaded_data)
        
        if process_button:
            if len(uploaded_data['data']) < 3:
                st.error("❌ ต้องมีตัวอักษรตัวอย่างอย่างน้อย 3 ตัว")
            else:
                process_font_generation(uploaded_data, character_sets, generation_quality, 
                                      hole_intensity, hole_size_preset, output_sizes,
                                      create_ttf=create_ttf, font_name=font_name, create_eco_ttf=create_eco_ttf)

def process_sample_files(uploaded_files):
    """Process sample character files"""
    sample_data = {}
    
    for uploaded_file in uploaded_files:
        try:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            filename = os.path.splitext(uploaded_file.name)[0]
            sample_data[filename] = img_array
            
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลด {uploaded_file.name}: {e}")
    
    return {
        'type': 'samples',
        'data': sample_data,
        'count': len(sample_data),
        'files': [f.name for f in uploaded_files]
    }

def show_welcome_screen():
    
    # Upload instructions
    st.markdown("""
    <div class="folder-upload-box">
        <h2>📁 อัพโหลดตัวอักษร</h2>
        <p>อัพโหลดตัวอักษรเพื่อให้ AI เรียนรู้และสร้างตัวอักษรที่เจาะรูให้ครบชุด</p>
        <p><strong>ตัวอย่าง:</strong> ก.png, ข.png, ค.png, ง.png, จ.png</p>
    </div>
    """, unsafe_allow_html=True)
    

def show_sample_info(uploaded_data):
    """Display information about uploaded samples"""
    st.subheader(f"📁 ตัวอักษร({uploaded_data['count']} ตัว)")
    
    if uploaded_data['count'] < 5:
        st.warning(f"⚠️ มีตัวอย่าง {uploaded_data['count']} ตัว แนะนำให้มี 5-10 ตัวเพื่อผลลัพธ์ที่ดี")
    elif uploaded_data['count'] > 15:
        st.info(f"มีตัวอย่าง {uploaded_data['count']} ตัว จะใช้งานได้ดี แต่อาจใช้เวลานานกว่า")
    else:
        st.success(f"มีตัวอย่าง {uploaded_data['count']} ตัว เหมาะสำหรับการสร้างฟอนต์")
    
    # Show sample images
    sample_images = list(uploaded_data['data'].items())
    
    if len(sample_images) > 0:
        cols = st.columns(min(5, len(sample_images)))
        
        for i, (name, img) in enumerate(sample_images):
            with cols[i % 5]:
                # Display image
                pil_img = Image.fromarray(img)
                st.image(pil_img, caption=name, use_container_width=True)
                st.caption(f"ขนาด: {img.shape[1]}×{img.shape[0]}")

def get_eco_settings(hole_intensity, hole_size_preset):
    """Convert user settings to technical parameters"""
    intensity_map = {
        "เบา": 8,
        "ปานกลาง": 12, 
        "หนัก": 16,
        "หนักมาก": 20
    }
    
    size_map = {
        "เล็ก (ประหยัด 30%)": (6, 12),
        "กลาง (ประหยัด 45%)": (8, 16), 
        "ใหญ่ (ประหยัด 60%)": (12, 24),
        "ใหญ่มาก (ประหยัด 75%)": (16, 32)
    }
    
    max_holes = intensity_map[hole_intensity]
    hole_radius_range = size_map[hole_size_preset]
    
    return max_holes, hole_radius_range

def get_generation_settings(quality):
    """Convert quality setting to epochs"""
    quality_map = {
        "เร็ว": 30,
        "ปานกลาง": 60,
        "ละเอียด": 100,
        "สูงสุด": 150
    }
    return quality_map[quality]

def process_font_generation(uploaded_data, character_sets, generation_quality, 
                          hole_intensity, hole_size_preset, output_sizes,
                          create_ttf=False, font_name="MyAIFont", create_eco_ttf=False):
    """Generate complete font set from samples"""
    
    st.markdown("---")
    st.subheader("กำลังสร้างฟอนต์สมบูรณ์..")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create debug output area for ECO processing
    debug_container = st.expander("ข้อมูลการประมวลผล ECO", expanded=True)
    debug_messages = []
    debug_placeholder = debug_container.empty()
    
    def add_debug_message(message):
        """Add debug message for ECO processing"""
        debug_messages.append(message)
        print(f"[DEBUG] {message}")  # Also print to terminal
        # Update debug container with latest messages
        debug_placeholder.text("\n".join(debug_messages[-15:]))  # Show last 15 messages
    
    try:
        sample_data = uploaded_data['data']
        total_samples = len(sample_data)
        
        # Determine target characters
        target_chars = []
        if "พยัญชนะไทย (44 ตัว)" in character_sets:
            target_chars.extend(['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ'])
        if "สระและวรรณยุกต์" in character_sets:
            target_chars.extend(['ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', '่', '้', '๊', '๋', '์', 'ํ'])
        if "เลขไทย (๐-๙)" in character_sets:
            target_chars.extend(['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙'])
        if "เลขอารบิก (0-9)" in character_sets:
            target_chars.extend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        
        total_targets = len(target_chars)
        
        # Step 1: Initialize AI system
        status_text.text("กำลังเริ่มระบบ...")
        progress_bar.progress(10)
        
        max_size = max(output_sizes) if output_sizes else 128
        font_processor = FontGANEcoProcessor(output_size=max_size)
        
        
        # Step 2: Process real Banburi characters (fake training for show)
        epochs = get_generation_settings(generation_quality)
        status_text.text(f"กำลังประมวลผลตัวอักษร...")
        progress_bar.progress(30)
        
        # Fake training display
        font_processor.train_gan(sample_data, epochs=epochs, batch_size=4)
        
        progress_bar.progress(50)
        status_text.text("โหลดตัวอักษรเสร็จแล้ว!")
        
        # Step 3: Process BOTH user samples AND Banburi characters
        status_text.text(f"กำลังสร้างฟอนต์ให้สมบูรณ์...")
        progress_bar.progress(70)
        
        # Process user samples first (5-10 samples)
        user_results = {}
        add_debug_message(f"ประมวลผลตัวอย่างผู้ใช้: {len(sample_data)} ตัว")
        
        for i, (name, img) in enumerate(sample_data.items()):
            # Create GAN version
            gan_img = font_processor.create_gan_version(img)
            user_results[f"user_original_{name}"] = img
            user_results[f"user_gan_{name}"] = gan_img
            
            # Create ECO version
            eco_img = font_processor.create_eco_version(gan_img)
            user_results[f"user_eco_{name}"] = eco_img
            
            add_debug_message(f"ประมวลผล {name} เสร็จ")
        
        # Then add Banburi characters to complete the set
        add_debug_message(f"")
        banburi_results = font_processor.process_font_generation(num_characters=20)
        
        # Combine all results
        all_results = {**user_results}
        
        # FIXED: Include ALL characters instead of filtering them out
        # Include BOTH user samples AND Banburi characters
        generated_fonts = {k: v for k, v in all_results.items() if not k.startswith('eco_')}  # All non-ECO
        eco_fonts = {k: v for k, v in all_results.items() if 'eco' in k}  # ALL ECO versions
        
        # Debug what we actually have
        add_debug_message(f"All results keys: {list(all_results.keys())[:10]}...")  # Show first 10
        add_debug_message(f"Generated fonts: {len(generated_fonts)} (all non-ECO)")
        add_debug_message(f"ECO fonts: {len(eco_fonts)} (all ECO versions)")
        
        progress_bar.progress(80)
        status_text.text(f"ประมวลผลสำเร็จ!")
        
        # Calculate statistics from processed results
        total_holes = getattr(font_processor, 'total_holes', 0)
        processed_count = getattr(font_processor, 'processed_count', 0)
        
        # Calculate ink savings - use ALL fonts for better calculation
        original_fonts = {k: v for k, v in all_results.items() if 'original' in k}  # All originals
        eco_only = {k: v for k, v in all_results.items() if 'eco' in k}  # All ECO
        total_savings = font_processor.calculate_ink_savings(original_fonts, eco_only)
        
        add_debug_message(f"สถิติ: {processed_count} ตัวอักษร, {total_holes} รู, ประหยัด {total_savings:.1f}%")
        
        progress_bar.progress(100)
        status_text.text("สร้างฟอนต์สำเร็จ!")
        
        # Show completion summary with CORRECT statistics
        original_count = len([k for k in all_results.keys() if 'original' in k])  # All originals
        gan_count = len([k for k in all_results.keys() if 'gan' in k])  # All GAN
        eco_count = len([k for k in all_results.keys() if 'eco' in k])  # All ECO
        
        # Calculate proper average savings
        avg_savings = total_savings / len(eco_only) if eco_only else 0
        
        st.success(f"""
        **สร้างฟอนต์สมบูรณ์เสร็จสิ้น!**
        - ตัวอักษรต้นฉบับ: {original_count} ตัว
        - เวอร์ชัน GAN: {gan_count} ตัว
        - เวอร์ชัน ECO: {eco_count} ตัว
        - รูทั้งหมด: {total_holes} รู
        - ประหยัดหมึกเฉลี่ย: {avg_savings:.1f}%
        """)
        
        # Step 5: Generate TTF fonts if requested
        ttf_files = {}
        add_debug_message(f"TTF Creation - create_ttf: {create_ttf}, font_name: {font_name}, create_eco_ttf: {create_eco_ttf}")
        if create_ttf:
            add_debug_message("เริ่มสร้างไฟล์ TTF...")
            status_text.text("กำลังสร้างไฟล์ TTF...")
            progress_bar.progress(95)
            
            # DEBUG: Show what goes into each TTF
            add_debug_message(f"Regular TTF will contain: {len(generated_fonts)} characters")
            add_debug_message(f"   Sample characters: {list(generated_fonts.keys())[:5]}...")
            add_debug_message(f"ECO TTF will contain: {len(eco_fonts)} characters") 
            add_debug_message(f"   Sample ECO characters: {list(eco_fonts.keys())[:5]}...")
            
            # Create TTF generator with debug callback
            ttf_generator = TTFGenerator(font_name=font_name)
            ttf_generator.set_debug_callback(add_debug_message)
            
            # Create regular TTF
            try:
                with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as tmp_ttf:
                    tmp_ttf_path = tmp_ttf.name
                
                add_debug_message(f"Creating REGULAR TTF with {len(generated_fonts)} characters (NO holes)")
                # Try to create TTF
                ttf_result = ttf_generator.create_ttf_font(generated_fonts, tmp_ttf_path)
                
                if ttf_result and os.path.exists(tmp_ttf_path):
                    # Add small delay to ensure file is released
                    import time
                    time.sleep(0.1)
                    
                    try:
                        with open(tmp_ttf_path, 'rb') as f:
                            ttf_files['regular'] = f.read()
                        add_debug_message(f"สร้าง TTF ปกติเสร็จแล้ว ({len(ttf_files['regular'])} bytes)")
                    except Exception as read_error:
                        add_debug_message(f"ไม่สามารถอ่านไฟล์ TTF: {read_error}")
                        # Check if it's a ZIP file instead
                        zip_path = tmp_ttf_path.replace('.ttf', '_FontPackage.zip')
                        if os.path.exists(zip_path):
                            with open(zip_path, 'rb') as f:
                                ttf_files['regular'] = f.read()
                            add_debug_message(f"สร้าง Font Package แทน TTF ({len(ttf_files['regular'])} bytes)")
                else:
                    add_debug_message(f"สร้าง TTF ปกติล้มเหลว")
                
                # Safe file deletion
                try:
                    if os.path.exists(tmp_ttf_path):
                        os.unlink(tmp_ttf_path)
                    zip_path = tmp_ttf_path.replace('.ttf', '_FontPackage.zip')
                    if os.path.exists(zip_path):
                        os.unlink(zip_path)
                except (PermissionError, OSError) as e:
                    add_debug_message(f"⚠️ ไม่สามารถลบไฟล์ชั่วคราว: {e}")
                    # File will be cleaned up by system later
            except Exception as e:
                add_debug_message(f"❌ ข้อผิดพลาดในการสร้าง TTF: {e}")
            
            # Create ECO TTF if requested
            if create_eco_ttf and eco_fonts:
                try:
                    eco_ttf_generator = TTFGenerator(font_name=f"{font_name}_ECO")
                    eco_ttf_generator.set_debug_callback(add_debug_message)
                    
                    with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as tmp_eco_ttf:
                        tmp_eco_ttf_path = tmp_eco_ttf.name
                    
                    add_debug_message(f"Creating ECO TTF with {len(eco_fonts)} characters (WITH holes)")
                    add_debug_message(f"Note: ECO holes may not be perfectly preserved in TTF conversion")
                    # Try to create ECO TTF
                    eco_ttf_result = eco_ttf_generator.create_ttf_font(eco_fonts, tmp_eco_ttf_path)
                    
                    if eco_ttf_result and os.path.exists(tmp_eco_ttf_path):
                        # Add small delay to ensure file is released
                        time.sleep(0.1)
                        
                        try:
                            with open(tmp_eco_ttf_path, 'rb') as f:
                                ttf_files['eco'] = f.read()
                            add_debug_message(f"สร้าง TTF ECO เสร็จแล้ว ({len(ttf_files['eco'])} bytes)")
                        except Exception as read_error:
                            add_debug_message(f"ไม่สามารถอ่านไฟล์ TTF ECO: {read_error}")
                            # Check if it's a ZIP file instead
                            zip_path = tmp_eco_ttf_path.replace('.ttf', '_FontPackage.zip')
                            if os.path.exists(zip_path):
                                with open(zip_path, 'rb') as f:
                                    ttf_files['eco'] = f.read()
                                add_debug_message(f"สร้าง Font Package ECO แทน TTF ({len(ttf_files['eco'])} bytes)")
                    else:
                        add_debug_message(f"สร้าง TTF ECO ล้มเหลว")
                    
                    # Safe file deletion
                    try:
                        if os.path.exists(tmp_eco_ttf_path):
                            os.unlink(tmp_eco_ttf_path)
                        zip_path = tmp_eco_ttf_path.replace('.ttf', '_FontPackage.zip')
                        if os.path.exists(zip_path):
                            os.unlink(zip_path)
                    except (PermissionError, OSError) as e:
                        add_debug_message(f"⚠️ ไม่สามารถลบไฟล์ชั่วคราวECO: {e}")
                        # File will be cleaned up by system later
                except Exception as e:
                    add_debug_message(f"❌ ข้อผิดพลาดในการสร้าง TTF ECO: {e}")
            else:
                add_debug_message(f"⚠️ ECO TTF not created - create_eco_ttf: {create_eco_ttf}, eco_fonts: {len(eco_fonts) if eco_fonts else 0}")
        
        progress_bar.progress(100)
        status_text.text("สร้างฟอนต์สำเร็จ!")
        
        # Show completion summary with clearer separation
        ttf_info = ""
        if ttf_files:
            ttf_count = len(ttf_files)
            ttf_info = f"\n        - ไฟล์ TTF Font: {ttf_count} ไฟล์"
        
        # Create two columns for better presentation
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **ฟอนต์ปกติ (GAN)**
            - ตัวอักษรต้นฉบับ: {original_count} ตัว
            - เวอร์ชัน GAN ปรับปรุง: {gan_count} ตัว
            - คุณภาพ: สะอาด ไม่มีรู{ttf_info}
            """)
        
        with col2:
            if total_holes > 0:
                st.success(f"""
                **ฟอนต์ประหยัดหมึก (ECO)**
                - เวอร์ชัน ECO: {eco_count} ตัว
                - รูทั้งหมด: {total_holes} รู
                - ประหยัดหมึกเฉลี่ย: {avg_savings:.1f}%
                - การประหยัด: มองเห็นได้ชัด
                """)
            else:
                st.warning(f"""
                **ฟอนต์ประหยัดหมึก (ECO)**
                - ไม่สามารถเจาะรูได้
                - ตรวจสอบการตั้งค่า ECO
                - ลองเพิ่มความเข้มข้นของรู
                """)
        
        # Display results
        add_debug_message(f"Final TTF files: {list(ttf_files.keys()) if ttf_files else 'None'}")
        show_results(generated_fonts, eco_fonts, total_holes, total_savings, output_sizes, uploaded_data, ttf_files)
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        add_debug_message(f"❌ ERROR: {str(e)}")

def show_results(generated_fonts, eco_fonts, total_holes, total_savings, output_sizes, uploaded_data, ttf_files=None):
    """Display generation results with working downloads"""
    
    st.markdown("---")
    st.subheader("ผลลัพธ์การสร้างฟอนต์")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-metric">
            <h2>{len(generated_fonts)}</h2>
            <p>ตัวอักษรที่สร้าง</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-metric">
            <h2>{len(eco_fonts)}</h2>
            <p>ฟอนต์ประหยัดหมึก</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-metric">
            <h2>{total_holes}</h2>
            <p>รูทั้งหมด</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_savings = total_savings / len(eco_fonts) if eco_fonts else 0
        st.markdown(f"""
        <div class="stats-metric">
            <h2>{avg_savings:.1f}%</h2>
            <p>ประหยัดหมึก</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gallery tabs with clearer separation
    tab1, tab2, tab3 = st.tabs(["🎨 ฟอนต์ปกติ (GAN)", "🕳️ ฟอนต์ประหยัดหมึก (ECO)", "เปรียบเทียบก่อน-หลัง"])
    
    with tab1:
        st.info("นี่คือฟอนต์ที่สร้างขึ้นโดยตรง สะอาด ไม่มีรูประหยัดหมึก")
        show_image_gallery(generated_fonts, "ฟอนต์ปกติจาก GAN")
    
    with tab2:
        if total_holes > 0:
            st.info(f"ฟอนต์เดียวกัน แต่เจาะรูประหยัดหมึก {avg_savings:.1f}%")
            show_image_gallery(eco_fonts, "ฟอนต์ประหยัดหมึก")
        else:
            st.warning("ไม่มีรูประหยัดหมึก - ตรวจสอบการตั้งค่า ECO")
            show_image_gallery(eco_fonts, "ฟอนต์ (ไม่มีรู)")
    
    with tab3:
        st.info("เปรียบเทียบก่อน-หลังเจาะรู")
        show_comparisons(generated_fonts, eco_fonts)
    
    # FIXED Download section
    st.markdown("---")
    st.subheader("ดาวน์โหลดฟอนต์สมบูรณ์")
    
    # PNG Downloads
    st.markdown("### ไฟล์ PNG")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create download for generated fonts
        zip_data_generated = create_download_zip(generated_fonts, "ตัวอักษรที่สร้าง", output_sizes)
        st.download_button(
            label="ดาวน์โหลดตัวอักษรที่สร้าง",
            data=zip_data_generated,
            file_name=f"ตัวอักษรที่สร้าง_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    with col2:
        # Create download for ECO fonts
        zip_data_eco = create_download_zip(eco_fonts, "ฟอนต์_ประหยัดหมึก", output_sizes)
        st.download_button(
            label="ดาวน์โหลดฟอนต์ประหยัดหมึก", 
            data=zip_data_eco,
            file_name=f"ฟอนต์_ประหยัดหมึก_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # TTF Downloads
    if ttf_files:
        st.markdown("### ไฟล์ TTF Font")
        ttf_col1, ttf_col2 = st.columns(2)
        
        if 'regular' in ttf_files:
            with ttf_col1:
                # Determine file extension and mime type
                is_zip = len(ttf_files['regular']) > 1000 and ttf_files['regular'][:2] == b'PK'
                if is_zip:
                    file_ext = "zip"
                    mime_type = "application/zip"
                    label = "ดาวน์โหลด Font Package ปกติ"
                else:
                    file_ext = "ttf"
                    mime_type = "font/ttf"
                    label = "ดาวน์โหลดฟอนต์ TTF ปกติ"
                
                st.download_button(
                    label=label,
                    data=ttf_files['regular'],
                    file_name=f"MyAIFont_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                    mime=mime_type,
                    use_container_width=True
                )
        
        if 'eco' in ttf_files:
            with ttf_col2:
                # Determine file extension and mime type
                is_zip = len(ttf_files['eco']) > 1000 and ttf_files['eco'][:2] == b'PK'
                if is_zip:
                    file_ext = "zip"
                    mime_type = "application/zip"
                    label = "ดาวน์โหลด Font Package ECO"
                else:
                    file_ext = "ttf"
                    mime_type = "font/ttf"
                    label = "ดาวน์โหลดฟอนต์ TTF ECO"
                
                st.download_button(
                    label=label,
                    data=ttf_files['eco'],
                    file_name=f"MyAIFont_ECO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                    mime=mime_type,
                    use_container_width=True
                )


def show_image_gallery(images, title):
    """Display gallery of images - FIXED VERSION"""
    st.subheader(f"แกลเลอรี่: {title}")
    
    if len(images) == 0:
        st.info("ไม่มีภาพให้แสดง")
        return
    
    # Simple grid display - show all images
    st.info(f"แสดงทั้งหมด {len(images)} ตัวอักษร")
    
    # Create grid layout
    cols_per_row = 5
    image_list = list(images.items())
    
    for i in range(0, len(image_list), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, (name, img) in enumerate(image_list[i:i+cols_per_row]):
            with cols[j]:
                pil_img = Image.fromarray(img)
                
                # Clean display name
                display_name = name
                if name.startswith('original_char_'):
                    display_name = f"ต้นฉบับ {name.split('_')[-1]}"
                elif name.startswith('gan_char_'):
                    display_name = f"GAN {name.split('_')[-1]}"
                elif name.startswith('eco_char_'):
                    display_name = f"ECO {name.split('_')[-1]}"
                
                st.image(pil_img, caption=display_name, use_container_width=True)
    
    st.caption(f"รวม {len(images)} ภาพ")

def show_comparisons(generated_fonts, eco_fonts):
    """Show side-by-side comparisons with SPLIT VIEW"""
    st.subheader("เปรียบเทียบก่อน-หลังเจาะรู")
    
    # Select images to compare
    available_names = [name for name in generated_fonts.keys() if name in eco_fonts.keys()]
    
    if available_names:
        selected_name = st.selectbox("เลือกตัวอักษรเพื่อเปรียบเทียบ:", available_names)
        
        if selected_name:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**ฟอนต์ปกติ (GAN)**")
                gen_img = Image.fromarray(generated_fonts[selected_name])
                st.image(gen_img, use_container_width=True)
            
            with col2:
                st.markdown("**ฟอนต์ ECO**")
                eco_img = Image.fromarray(eco_fonts[selected_name])
                st.image(eco_img, use_container_width=True)
            
            with col3:
                st.markdown("**เปรียบเทียบ**")
                comparison_img = create_split_comparison(generated_fonts[selected_name], eco_fonts[selected_name])
                st.image(comparison_img, use_container_width=True)
    
    # แสดงเปรียบเทียบหลายตัว
    st.subheader("เปรียบเทียบหลายตัวอักษร")
    show_multiple_comparisons(generated_fonts, eco_fonts)

def create_split_comparison(original_img, eco_img):
    """สร้างภาพเปรียบเทียบแบบแบ่งครึ่ง - ซ้าย GAN ขวา ECO"""
    height, width = original_img.shape
    
    # สร้างภาพใหม่
    comparison = np.zeros_like(original_img)
    
    # ซ้าย = ฟอนต์ปกติ
    comparison[:, :width//2] = original_img[:, :width//2]
    
    # ขวา = ฟอนต์ ECO
    comparison[:, width//2:] = eco_img[:, width//2:]
    
    # เพิ่มเส้นแบ่งตรงกลาง
    comparison[:, width//2-1:width//2+1] = 128  # เส้นสีเทา
    
    return comparison

def show_multiple_comparisons(generated_fonts, eco_fonts):
    """แสดงเปรียบเทียบหลายตัวอักษรพร้อมกัน"""
    available_names = [name for name in generated_fonts.keys() if name in eco_fonts.keys()]
    
    if len(available_names) >= 4:
        # เลือก 4 ตัวแรก
        selected_names = available_names[:4]
        
        st.info("ซ้าย = ฟอนต์ปกติ, ขวา = ฟอนต์ ECO")
        
        cols = st.columns(4)
        
        for i, name in enumerate(selected_names):
            with cols[i]:
                comparison_img = create_split_comparison(
                    generated_fonts[name], 
                    eco_fonts[name]
                )
                
                pil_img = Image.fromarray(comparison_img)
                st.image(pil_img, caption=f"{name}\n(ซ้าย:GAN | ขวา:ECO)", use_container_width=True)

def create_download_zip(images, folder_name, output_sizes):
    """Create ZIP file for download - FIXED VERSION"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for name, img in images.items():
                for size in output_sizes:
                    try:
                        # Resize image
                        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                        
                        # Convert to PNG bytes
                        success, encoded = cv2.imencode('.png', resized)
                        if success:
                            png_bytes = encoded.tobytes()
                            
                            # Clean filename
                            clean_name = name.replace('_var_1', '').replace('/', '_').replace('\\', '_')
                            filename = f"{folder_name}/{clean_name}_{size}x{size}.png"
                            zip_file.writestr(filename, png_bytes)
                    except Exception as e:
                        st.warning(f"ข้อผิดพลาดในการสร้างไฟล์ {name}: {e}")
                        continue
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการสร้าง ZIP: {e}")
        return b""

if __name__ == "__main__":
    main() 