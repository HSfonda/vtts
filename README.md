# 🎙️ vtts - Fast Voice Synthesis on Your GPU

[![Download vtts](https://img.shields.io/badge/Download-vtts-brightgreen?style=for-the-badge)](https://github.com/HSfonda/vtts)

---

## 📋 What is vtts?

vtts is a program that turns text into speech. It uses your computer’s GPU to handle many voice requests at once. Think of it as a tool that speaks text fast and clearly, all on your local machine. vtts can work on over 10 requests at the same time using one GPU. It helps create voices for apps, assistants, or any project needing speech.

This application runs on Windows. You do not need to code or know deep technical details to use it. Just follow the simple steps below.

---

## 💻 System Requirements

- **Operating System:** Windows 10 or later (64-bit)
- **GPU:** NVIDIA graphics card with at least 4GB VRAM (e.g., GTX 1050, RTX 2060, or better)
- **CPU:** Quad-core processor or better
- **Memory:** Minimum 8GB RAM
- **Disk Space:** At least 500 MB free
- **Internet:** Required for initial download and setup only

---

## 🚀 Getting Started: Download and Install vtts

1. Click the big green button below to visit the download page:

   [![Download vtts](https://img.shields.io/badge/Download-vtts-blue?style=for-the-badge)](https://github.com/HSfonda/vtts)

2. On the GitHub page, look for the latest release section.

3. Download the Windows installer file (usually named `vtts-setup.exe` or similar).

4. Once the file downloads, open it by double-clicking.

5. Follow the setup instructions:
   - Click **Next** when prompted.
   - Agree to the license terms.
   - Choose an installation folder or use the default.
   - Click **Install**.

6. Wait a few moments while vtts installs.

7. When finished, click **Finish** to close the installer.

---

## 🛠️ Setting Up and Running vtts

1. Find the **vtts** icon on your desktop or in your Start menu.

2. Double-click the icon to open the application.

3. The main window shows a simple box where you can type any text.

4. Type your text message in the box.

5. Press the **Speak** button.

6. vtts will convert the text and play it aloud.

7. You can enter new texts and use the Speak button as many times as you want.

---

## ⚙️ Features of vtts

- Processes many speech requests at once, so you don’t wait.
- Uses your GPU to speed up voice creation.
- Supports natural-sounding voice models.
- Real-time playback: hear speech immediately.
- Works without internet after setup.
- Works with different voice styles.
- Simple interface, no coding required.

---

## 🎯 How vtts Uses Your GPU

vtts takes advantage of your NVIDIA graphics card to speed up its work. Most TTS programs only use your computer’s CPU (processor). vtts uses the GPU to handle complex calculations faster. This means it can create multiple speech streams at once without slowing down. 

Your GPU must have the right drivers installed. You can get these from the NVIDIA website. If you want smooth performance, install the latest driver for your card.

---

## 🖥️ Running vtts From the Command Line (Optional)

For users who want some control over how vtts runs, there is a simple command line option.

1. Open the **Command Prompt**:
   - Press **Windows + R**, type `cmd`, and press Enter.

2. Navigate to the folder where vtts is installed:
   ```
   cd "C:\Program Files\vtts"
   ```

3. Run vtts using this command:
   ```
   vtts.exe
   ```

This will start the app as usual.

---

## 🔧 Troubleshooting

- **No sound or speech doesn’t play**
  - Check if your speakers or headphones are connected and working.
  - Make sure your system volume is not muted.
  - Restart vtts and try again.

- **vtts won’t start**
  - Check your antivirus or firewall is not blocking the app.
  - Restart your computer.
  - Make sure you installed the latest version from the download page.

- **GPU not detected or slow performance**
  - Verify your NVIDIA driver is installed and up to date.
  - Check your graphics card meets the minimum 4GB VRAM requirement.
  - Close other heavy programs that might use the GPU.

---

## 🔄 Updating vtts

To get the newest features or fixes:

1. Visit the [vtts GitHub page](https://github.com/HSfonda/vtts).

2. Download the latest version from the releases section.

3. Run the installer — it will replace the old version while keeping your settings.

---

## ⚡ How to Get Help

- Check the Issues section on the GitHub page for solutions to common problems.
- Use the Discussions tab to ask questions.
- Look for FAQ or Wiki pages linked on the repository, if available.

---

## 📂 More About vtts

This application specializes in real-time voice synthesis. It does not require internet after you install it. The program uses PyTorch and specialized TTS models optimized for GPU performance. It’s designed to handle multiple users or apps asking for speech from the same machine.

vtts supports continuous batching. This means the program groups speech requests together to work more efficiently. It fits small and large voice projects and can be used for voice agents, voice cloning, and other voice synthesis tasks.

---

[![Download vtts](https://img.shields.io/badge/Download-vtts-brightgreen?style=for-the-badge)](https://github.com/HSfonda/vtts)