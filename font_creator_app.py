import streamlit as st
import os
import tempfile
from PIL import Image
import cv2
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder

def image_to_contours(img_array, output_size=1000):
    """
    Converts a single image array to a list of contours.
    Flips the contours vertically for correct font orientation.
    """
    # Convert PIL/numpy array to OpenCV format
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    # Threshold to get a binary image (black character on white background)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []

    # Get bounding box to scale and position correctly
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Scale contours to fit within the font's em square
    scale = output_size * 0.8 / max(w, h)  # Use 80% of em square
    
    scaled_contours = []
    for contour in contours:
        # Scale and center
        scaled_contour = (contour - [x, y]) * scale
        # Center horizontally and vertically
        scaled_contour[:, :, 0] += (output_size - w * scale) / 2
        scaled_contour[:, :, 1] += (output_size - h * scale) / 4  # Slight offset from bottom
        # Flip vertically (font coordinates have Y going up)
        scaled_contour[:, :, 1] = output_size - scaled_contour[:, :, 1]
        scaled_contours.append(scaled_contour.astype(int))
        
    return scaled_contours

def contours_to_glyph(contours):
    """Converts a list of contours to a fontTools Glyph object."""
    if not contours:
        return None

    pen = TTGlyphPen(None)
    for contour in contours:
        if len(contour) < 3:  # Skip invalid contours
            continue
            
        # Start the path
        pen.moveTo(tuple(contour[0][0]))
        # Add lines for the rest of the points
        for point in contour[1:]:
            pen.lineTo(tuple(point[0]))
        # Close the path
        pen.closePath()
        
    return pen.glyph()

def create_notdef_glyph():
    """Create a .notdef glyph (empty rectangle)"""
    pen = TTGlyphPen(None)
    # Draw empty rectangle
    pen.moveTo((50, 50))
    pen.lineTo((450, 50))
    pen.lineTo((450, 650))
    pen.lineTo((50, 650))
    pen.closePath()
    return pen.glyph()

def create_ttf_from_images(image_char_map, font_name, output_path):
    """
    Creates a TTF font from a dictionary mapping image arrays to characters.
    """
    # Font metrics
    upm = 1000  # Units Per Em
    ascender = 800
    descender = -200
    glyph_width = 600

    # Create FontBuilder instance
    fb = FontBuilder(upm, isTTF=True)
    
    # Prepare glyph names and character mapping
    glyph_names = [".notdef"] + [f"char_{ord(char):04X}" for char in image_char_map.values()]
    char_map = {ord(char): f"char_{ord(char):04X}" for char in image_char_map.values()}
    
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap(char_map)
    
    # Create glyphs
    glyphs = {}
    
    # Create .notdef glyph (required)
    glyphs[".notdef"] = create_notdef_glyph()
    
    # Create character glyphs
    for img_array, char in image_char_map.items():
        glyph_name = f"char_{ord(char):04X}"
        contours = image_to_contours(img_array, upm)
        
        if contours:
            glyph = contours_to_glyph(contours)
            if glyph:
                glyphs[glyph_name] = glyph
            else:
                # Create empty glyph if contour processing fails
                glyphs[glyph_name] = TTGlyphPen(None).glyph()
        else:
            # Create empty glyph if no contours found
            glyphs[glyph_name] = TTGlyphPen(None).glyph()

    fb.setupGlyf(glyphs)

    # Setup metrics
    metrics = {glyph_name: (glyph_width, 50) for glyph_name in glyph_names}
    fb.setupHorizontalMetrics(metrics)

    # Font metadata
    fb.setupNameTable({
        "familyName": font_name,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"1.0; {font_name}",
        "fullName": f"{font_name} Regular",
        "psName": font_name.replace(" ", ""),
        "version": "1.0"
    })
    
    fb.setupHorizontalHeader(ascent=ascender, descent=descender)
    fb.setupOS2()
    fb.setupPost()
    
    # Save the font
    font = fb.font
    font.save(output_path)

# Streamlit UI
st.set_page_config(
    page_title="🎨 PNG to TTF Font Creator",
    page_icon="🔤",
    layout="wide"
)

st.title("🎨 PNG to TTF Font Creator")
st.markdown("### สร้างฟอนต์ TTF จากภาพ PNG โดยใช้ Pure Python!")

st.info("💡 วิธีใหม่นี้ใช้ OpenCV + FontTools ไม่ต้องติดตั้งโปรแกรมภายนอก")

font_name = st.text_input("🏷️ ชื่อฟอนต์:", "MyStreamlitFont")

uploaded_files = st.file_uploader(
    "📁 อัพโหลดภาพตัวอักษร (PNG)",
    type="png",
    accept_multiple_files=True,
    help="อัพโหลดภาพ PNG ที่มีตัวอักษรสีดำบนพื้นขาว"
)

if uploaded_files:
    st.success(f"📊 อัพโหลดแล้ว {len(uploaded_files)} ไฟล์")
    
    # Character mapping interface
    st.subheader("🔤 กำหนดตัวอักษรให้กับแต่ละภาพ")
    
    if 'char_map' not in st.session_state:
        st.session_state.char_map = {}

    cols = st.columns(min(4, len(uploaded_files)))
    
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, width=100)
            
            # Character input
            char = st.text_input(
                f"ตัวอักษรสำหรับ `{uploaded_file.name}`", 
                key=f"char_{uploaded_file.name}",
                max_chars=1,
                placeholder="เช่น ก, A, 1"
            )
            
            if char:
                # Convert image to numpy array
                img_array = np.array(image)
                st.session_state.char_map[char] = img_array

    # Show current mapping
    if st.session_state.char_map:
        st.subheader("📋 ตัวอักษรที่กำหนดแล้ว")
        mapped_chars = list(st.session_state.char_map.keys())
        st.write(f"**{len(mapped_chars)} ตัวอักษร:** {', '.join(mapped_chars)}")

    # Generate font button
    if st.button("🚀 สร้างไฟล์ฟอนต์ TTF", disabled=(len(st.session_state.char_map) == 0)):
        if len(st.session_state.char_map) == 0:
            st.error("❌ กรุณากำหนดตัวอักษรให้กับภาพอย่างน้อย 1 ภาพ")
        else:
            with st.spinner("🔄 กำลังประมวลผลภาพและสร้างฟอนต์..."):
                try:
                    output_filename = f"{font_name}.ttf"
                    
                    # Create font
                    # Flip the mapping: {img_array: char}
                    image_char_mapping = {tuple(img_array.flatten()): char 
                                        for char, img_array in st.session_state.char_map.items()}
                    
                    # Convert back to proper format
                    final_mapping = {}
                    for char, img_array in st.session_state.char_map.items():
                        final_mapping[img_array] = char
                    
                    create_ttf_from_images(final_mapping, font_name, output_filename)
                    
                    st.success(f"✅ สร้างฟอนต์ '{output_filename}' สำเร็จ!")
                    
                    # Check file size
                    if os.path.exists(output_filename):
                        file_size = os.path.getsize(output_filename)
                        st.info(f"📊 ขนาดไฟล์: {file_size:,} bytes")
                        
                        # Download button
                        with open(output_filename, "rb") as f:
                            st.download_button(
                                label="⬇️ ดาวน์โหลดไฟล์ TTF",
                                data=f.read(),
                                file_name=output_filename,
                                mime="font/ttf"
                            )
                    else:
                        st.error("❌ ไม่พบไฟล์ที่สร้าง")

                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                    st.exception(e)

else:
    # Welcome screen
    st.markdown("""
    ## 🌟 คุณสมบัติใหม่
    
    - ✅ **Pure Python** - ไม่ต้องติดตั้งโปรแกรมภายนอก
    - ✅ **OpenCV Vectorization** - แปลง PNG เป็น vector อัตโนมัติ
    - ✅ **FontTools Integration** - สร้าง TTF มาตรฐาน
    - ✅ **Real-time Preview** - ดูผลลัพธ์ทันที
    
    ## 📋 วิธีใช้งาน
    
    1. **อัพโหลดภาพ PNG** ที่มีตัวอักษรสีดำบนพื้นขาว
    2. **กำหนดตัวอักษร** ให้กับแต่ละภาพ
    3. **คลิกสร้างฟอนต์** และดาวน์โหลด TTF
    
    ## 💡 เทคนิคสำหรับผลลัพธ์ที่ดี
    
    - ใช้ภาพความละเอียดสูง (300+ pixels)
    - ตัวอักษรสีดำบนพื้นขาว
    - ขอบชัดเจน ไม่เบลอ
    - ไม่มีเงาหรือลายน้ำ
    """) 