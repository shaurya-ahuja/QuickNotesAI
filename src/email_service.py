"""
QuickNotes-AI Email Service
SMTP-based email sending for meeting summaries.
100% Local - No Data Leaves Your Device (except when sending emails)
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class EmailConfig:
    """SMTP email configuration."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    use_tls: bool = True


@dataclass
class EmailResult:
    """Result of email sending operation."""
    success: bool
    message: str
    recipients_sent: List[str] = None
    recipients_failed: List[str] = None
    
    def __post_init__(self):
        if self.recipients_sent is None:
            self.recipients_sent = []
        if self.recipients_failed is None:
            self.recipients_failed = []


class EmailService:
    """
    Email service for sending meeting summaries via SMTP.
    Supports Gmail and other SMTP providers.
    """
    
    # Common SMTP configurations
    SMTP_PRESETS = {
        "gmail": {
            "server": "smtp.gmail.com",
            "port": 587,
            "use_tls": True
        },
        "outlook": {
            "server": "smtp-mail.outlook.com",
            "port": 587,
            "use_tls": True
        },
        "yahoo": {
            "server": "smtp.mail.yahoo.com",
            "port": 587,
            "use_tls": True
        }
    }
    
    def __init__(self, config: EmailConfig = None):
        """
        Initialize email service.
        
        Args:
            config: Optional EmailConfig, can be set later.
        """
        self.config = config or EmailConfig()
    
    def configure(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        use_tls: bool = True
    ):
        """Configure SMTP settings."""
        self.config = EmailConfig(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            sender_password=sender_password,
            use_tls=use_tls
        )
    
    def configure_from_preset(
        self,
        preset: str,
        sender_email: str,
        sender_password: str
    ):
        """Configure using a preset (gmail, outlook, yahoo)."""
        if preset not in self.SMTP_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.SMTP_PRESETS.keys())}")
        
        preset_config = self.SMTP_PRESETS[preset]
        self.configure(
            smtp_server=preset_config["server"],
            smtp_port=preset_config["port"],
            sender_email=sender_email,
            sender_password=sender_password,
            use_tls=preset_config["use_tls"]
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(
            self.config.smtp_server and
            self.config.sender_email and
            self.config.sender_password
        )
    
    def send_summary(
        self,
        recipients: List[str],
        meeting_title: str,
        summary: str,
        action_items: List[str] = None,
        transcript: str = None,
        attachments: List[str] = None
    ) -> EmailResult:
        """
        Send a meeting summary email.
        
        Args:
            recipients: List of recipient email addresses.
            meeting_title: Title of the meeting.
            summary: Meeting summary text (bullet points).
            action_items: Optional list of action items.
            transcript: Optional full transcript.
            attachments: Optional list of file paths to attach.
            
        Returns:
            EmailResult with success status and details.
        """
        if not self.is_configured:
            return EmailResult(
                success=False,
                message="Email not configured. Please set up SMTP settings."
            )
        
        if not recipients:
            return EmailResult(
                success=False,
                message="No recipients specified."
            )
        
        # Build email content
        subject = f"📝 Meeting Notes: {meeting_title}"
        html_body = self._build_html_body(meeting_title, summary, action_items, transcript)
        text_body = self._build_text_body(meeting_title, summary, action_items, transcript)
        
        # Send to each recipient
        sent = []
        failed = []
        
        for recipient in recipients:
            try:
                self._send_email(
                    recipient=recipient,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    attachments=attachments
                )
                sent.append(recipient)
            except Exception as e:
                failed.append(recipient)
                print(f"Failed to send to {recipient}: {e}")
        
        if sent and not failed:
            return EmailResult(
                success=True,
                message=f"Email sent successfully to {len(sent)} recipient(s).",
                recipients_sent=sent
            )
        elif sent and failed:
            return EmailResult(
                success=True,
                message=f"Email sent to {len(sent)}, failed for {len(failed)}.",
                recipients_sent=sent,
                recipients_failed=failed
            )
        else:
            return EmailResult(
                success=False,
                message="Failed to send email to all recipients.",
                recipients_failed=failed
            )
    
    def _send_email(
        self,
        recipient: str,
        subject: str,
        html_body: str,
        text_body: str,
        attachments: List[str] = None
    ):
        """Send a single email."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.sender_email
        msg["To"] = recipient
        
        # Attach text and HTML versions
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Add attachments
        if attachments:
            for filepath in attachments:
                if os.path.exists(filepath):
                    self._attach_file(msg, filepath)
        
        # Send
        context = ssl.create_default_context()
        
        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
            if self.config.use_tls:
                server.starttls(context=context)
            server.login(self.config.sender_email, self.config.sender_password)
            server.send_message(msg)
    
    def _attach_file(self, msg: MIMEMultipart, filepath: str):
        """Attach a file to the email."""
        filename = os.path.basename(filepath)
        
        with open(filepath, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )
        msg.attach(part)
    
    def _build_html_body(
        self,
        meeting_title: str,
        summary: str,
        action_items: List[str] = None,
        transcript: str = None
    ) -> str:
        """Build HTML email body."""
        
        # Convert summary bullets to HTML
        summary_html = ""
        for line in summary.split("\n"):
            line = line.strip()
            if line.startswith(("•", "-", "*")):
                bullet = line.lstrip("•-* ")
                summary_html += f"<li>{bullet}</li>\n"
        
        # Action items HTML
        actions_html = ""
        if action_items:
            actions_html = "<h3>📋 Action Items</h3>\n<ul>\n"
            for item in action_items:
                actions_html += f"<li>{item}</li>\n"
            actions_html += "</ul>\n"
        
        # Transcript HTML
        transcript_html = ""
        if transcript:
            transcript_html = f"""
            <h3>📝 Full Transcript</h3>
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 12px;">
{transcript[:5000]}{"..." if len(transcript) > 5000 else ""}
            </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 2px solid #00d4aa;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #1a1a2e;
            margin-top: 25px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
        .badge {{
            display: inline-block;
            background-color: #00d4aa;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <h1>📝 {meeting_title} <span class="badge">QuickNotes-AI</span></h1>
    
    <h3>📌 Summary</h3>
    <ul>
        {summary_html}
    </ul>
    
    {actions_html}
    
    {transcript_html}
    
    <div class="footer">
        <p>Generated by <strong>QuickNotes-AI</strong> - 100% Local Meeting Notetaker</p>
        <p>🔒 Your data stays on your device</p>
    </div>
</body>
</html>
"""
        return html
    
    def _build_text_body(
        self,
        meeting_title: str,
        summary: str,
        action_items: List[str] = None,
        transcript: str = None
    ) -> str:
        """Build plain text email body."""
        
        text = f"""
📝 Meeting Notes: {meeting_title}
{'=' * 50}

📌 SUMMARY
{summary}

"""
        
        if action_items:
            text += "📋 ACTION ITEMS\n"
            for item in action_items:
                text += f"  • {item}\n"
            text += "\n"
        
        if transcript:
            text += f"""📝 FULL TRANSCRIPT
{'-' * 30}
{transcript[:5000]}{"..." if len(transcript) > 5000 else ""}

"""
        
        text += """
---
Generated by QuickNotes-AI - 100% Local Meeting Notetaker
🔒 Your data stays on your device
"""
        
        return text
    
    def test_connection(self) -> EmailResult:
        """Test SMTP connection without sending an email."""
        if not self.is_configured:
            return EmailResult(
                success=False,
                message="Email not configured."
            )
        
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
            
            return EmailResult(
                success=True,
                message="SMTP connection successful!"
            )
        except Exception as e:
            return EmailResult(
                success=False,
                message=f"Connection failed: {str(e)}"
            )


# Singleton instance
_service_instance: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get email service instance."""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = EmailService()
    
    return _service_instance
