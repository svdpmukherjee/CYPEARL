import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import certifi

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

# ============================================================
# DOMAIN INFRASTRUCTURE
# ============================================================
# Trusted: luxconsultancy.com, securenebula.com, lockgrid.com,
#          superfinance.com, trendyletter.com, greenenvi.com, wattvoltbridge.com
# Spoofed: secure-nebula.com, superfinance.org, lockgrid.net,
#          wattvoltbrdige.com, greenenvi.org, trendyletter.net, luxconsultancy.net,
#          securenebula.org
# ============================================================

# ============================================================
# FACTORIAL DESIGN
# ============================================================
# Generic (principal fraction I=ABCD): cells 1-8
# Role-matched (complementary fraction): cells 9-16

ROLE_MATCHED_CELLS = [
    # (cell, is_phishing, sender_familiarity, urgency, framing, phishing_quality)
    (9,  True,  "known_external",   "high", "reward", "high"),
    (10, True,  "unknown_external", "low",  "threat", "medium"),
    (11, True,  "known_external",   "low",  "threat", "medium"),
    (12, True,  "unknown_external", "high", "reward", "high"),
    (13, False, "known_external",   "low",  "reward", None),
    (14, False, "unknown_external", "high", "threat", None),
    (15, False, "known_external",   "high", "threat", None),
    (16, False, "unknown_external", "low",  "reward", None),
]


def build_cluster_emails(cluster, content_list):
    """Build 8 role-matched email dicts for a cluster from content_list.
    content_list: list of 8 dicts with keys: sn, se, su, bo
    """
    emails = []
    for i, (cell, is_phish, fam, urg, fram, pq) in enumerate(ROLE_MATCHED_CELLS):
        c = content_list[i]
        emails.append({
            "sender_name": c["sn"],
            "sender_email": c["se"],
            "subject": c["su"],
            "body": c["bo"],
            "is_phishing": is_phish,
            "order_id": 9 + i,
            "experimental": True,
            "factorial_category": {
                "type": "phishing" if is_phish else "legitimate",
                "sender": "known" if "known" in fam else "unknown",
                "urgency": urg,
                "framing": fram,
            },
            "email_type": "phishing" if is_phish else "legitimate",
            "sender_familiarity": fam,
            "urgency_level": urg,
            "framing_type": fram,
            "condition_type": "role_matched",
            "job_cluster": cluster,
            "factorial_cell": cell,
            "phishing_quality": pq,
        })
    return emails


# ============================================================
# WELCOME EMAIL (order_id=0, non-experimental)
# ============================================================
WELCOME_EMAIL = {
    "sender_name": "IRiSC Lab - University of Luxembourg",
    "sender_email": "irisc@uni.lu",
    "subject": "Study Instructions: Email Decision Making Simulation",
    "body": """
            <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; line-height: 1.7; color: #1f2937;">

                <!-- ========== YOUR ROLE ========== -->
                <div style="margin-bottom: 28px;">
                    <h3 style="color: #1e40af; font-size: 18px; margin: 0 0 12px 0; display: flex; align-items: center; gap: 10px;">
                        1. Your Role
                    </h3>
                    <p style="margin: 0; color: #374151;">
                        Imagine you work in the <strong style="color: #1e40af;">{{JOB_CLUSTER}}</strong> department at <strong style="color: #1e40af;">LuxConsultancy</strong>, a consulting firm based in <strong>Paris</strong> that serves clients around the world. As part of your role, you help keep the company's email system secure by reviewing incoming messages and deciding how to respond.
                    </p>
                    <p style="margin: 8px 0 0 0; color: #374151;">
                        {{ROLE_CONTEXT}}
                    </p>
                </div>

                <!-- ========== TRUSTED PARTNERS TABLE ========== -->
                <div style="margin-bottom: 28px;">
                    <h3 style="color: #1e40af; font-size: 18px; margin: 0 0 12px 0; display: flex; align-items: center; gap: 10px;">
                        2. Your Company & Trusted Partners
                    </h3>
                    <p style="margin: 0 0 16px 0; color: #374151;">
                        Below are your company and <strong>trusted business partners</strong> you regularly communicate with. Any domain <strong>NOT</strong> listed in the table is <strong>unknown</strong> to you.
                    </p>
                    <p style="
                        margin: 0 0 16px 0;
                        color: #374151;
                        padding: 8px 12px;
                        border-left: 4px solid #F59E0B;
                        background-color: #FFFBEB;
                        border-radius: 4px;
                        ">
                        💡 Emails from unknown senders aren't necessarily unsafe; some may be legitimate new business contacts, while others could be suspicious.
                    </p>

                    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                            <thead>
                                <tr style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);">
                                    <th style="padding: 14px 16px; text-align: left; color: white; font-weight: 600;">Company</th>
                                    <th style="padding: 14px 16px; text-align: left; color: white; font-weight: 600;">Relationship</th>
                                    <th style="padding: 14px 16px; text-align: left; color: white; font-weight: 600;">Trusted Domain</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="background: #eff6ff;">
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong style="color: #1e40af;">LuxConsultancy</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Your employer (internal)</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #dbeafe; padding: 4px 8px; border-radius: 4px; color: #1e40af; font-weight: 600;">@luxconsultancy.com</code></td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong>SecureNebula</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Cloud services provider</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@securenebula.com</code></td>
                                </tr>
                                <tr style="background: #f9fafb;">
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong>LockGrid</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Security solutions client</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@lockgrid.com</code></td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong>SuperFinance</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Tax & accounting partner</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@superfinance.com</code></td>
                                </tr>
                                <tr style="background: #f9fafb;">
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong>TrendyLetter</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Newsletter service</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@trendyletter.com</code></td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><strong>GreenEnvi</strong></td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">Environmental consulting</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@greenenvi.com</code></td>
                                </tr>
                                <tr style="background: #f9fafb;">
                                    <td style="padding: 12px 16px;"><strong>WattVoltBridge</strong></td>
                                    <td style="padding: 12px 16px;">Energy sector client</td>
                                    <td style="padding: 12px 16px;"><code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">@wattvoltbridge.com</code></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <p style="margin: 12px 0 0 0; color: #374151;">
                        <strong>Remember:</strong> When receiving any emails in your mailbox, you can return to this email anytime to check information by clicking on it in your inbox.
                    </p>
                </div>

                <!-- ========== WHAT YOU WILL DO ========== -->
                <div style="margin-bottom: 28px;">
                    <h3 style="color: #1e40af; font-size: 18px; margin: 0 0 16px 0; display: flex; align-items: center; gap: 10px;">
                        3. What You Will Do
                    </h3>
                    <p style="margin: 0 0 16px 0; color: #374151;">You will review <strong>16 workplace emails</strong> and decide how to respond. Use your judgment! For each email, choose one action:</p>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;">
                        <!-- Mark Safe - CheckCircle icon -->
                        <div style="background: #ecfdf5; padding: 14px 16px; border-radius: 10px; border: 1px solid #a7f3d0; text-align: center;">
                            <div style="display: flex; justify-content: center; margin-bottom: 8px;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="m9 12 2 2 4-4"></path></svg>
                            </div>
                            <div style="font-weight: 700; color: #065f46;">Mark Safe</div>
                            <div style="font-size: 12px; color: #047857;">Email looks legitimate</div>
                        </div>
                        <!-- Report - AlertOctagon icon -->
                        <div style="background: #fef2f2; padding: 14px 16px; border-radius: 10px; border: 1px solid #fecaca; text-align: center;">
                            <div style="display: flex; justify-content: center; margin-bottom: 8px;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" x2="12" y1="8" y2="12"></line><line x1="12" x2="12.01" y1="16" y2="16"></line></svg>
                            </div>
                            <div style="font-weight: 700; color: #991b1b;">Report</div>
                            <div style="font-size: 12px; color: #b91c1c;">Email seems suspicious</div>
                        </div>
                        <!-- Delete - Trash2 icon -->
                        <div style="background: #f3f4f6; padding: 14px 16px; border-radius: 10px; border: 1px solid #d1d5db; text-align: center;">
                            <div style="display: flex; justify-content: center; margin-bottom: 8px;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path><line x1="10" x2="10" y1="11" y2="17"></line><line x1="14" x2="14" y1="11" y2="17"></line></svg>
                            </div>
                            <div style="font-weight: 700; color: #374151;">Delete</div>
                            <div style="font-size: 12px; color: #6b7280;">Not needed or spam</div>
                        </div>
                        <!-- Ignore - Ban icon -->
                        <div style="background: #fefce8; padding: 14px 16px; border-radius: 10px; border: 1px solid #fef08a; text-align: center;">
                            <div style="display: flex; justify-content: center; margin-bottom: 8px;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#a16207" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="m4.9 4.9 14.2 14.2"></path></svg>
                            </div>
                            <div style="font-weight: 700; color: #854d0e;">Ignore</div>
                            <div style="font-size: 12px; color: #a16207;">Skip for now</div>
                        </div>
                    </div>
                </div>

                <!-- ========== COMPENSATION ========== -->
                <div style="margin-bottom: 28px;">
                    <h3 style="color: #1e40af; font-size: 18px; margin: 0 0 12px 0; display: flex; align-items: center; gap: 10px;">
                        4. Compensation
                    </h3>
                    <p style="margin: 0; color: #374151;">
                        This study takes approximately <strong>30 minutes</strong> to complete. You will receive <strong>£5</strong> as base compensation for your time and effort.
                    </p>
                </div>

                <!-- ========== HERO BONUS BANNER ========== -->
                <div style="background: linear-gradient(135deg, #059669 0%, #047857 50%, #065f46 100%); border-radius: 16px; padding: 0; margin-bottom: 28px; box-shadow: 0 10px 40px rgba(5, 150, 105, 0.4), 0 0 0 1px rgba(255,255,255,0.1) inset; overflow: hidden; position: relative;">
                    <!-- Animated shimmer effect overlay -->
                    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%); pointer-events: none;"></div>

                    <div style="padding: 24px 28px; text-align: center; position: relative;">
                        <h2 style="margin: 0 0 8px 0; color: #ffffff; font-size: 28px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">BONUS REWARDS</h2>
                        <p style="margin: 0; color: rgba(255,255,255,0.9); font-size: 15px;">Earn extra by clicking links in emails and making good judgments!</p>
                    </div>

                    <div style="background: rgba(255,255,255,0.95); padding: 24px 28px; display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px; max-width: 280px; background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%); padding: 20px; border-radius: 12px; border: 2px solid #10b981; text-align: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);">
                            <div style="width: 50px; height: 50px; background: #10b981; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                                <span style="font-size: 24px;">✓</span>
                            </div>
                            <div style="color: #065f46; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Per safe link clicked</div>
                            <div style="font-size: 36px; font-weight: 900; color: #059669; line-height: 1;">+5¢</div>
                            <div style="font-size: 12px; color: #047857; margin-top: 6px;">Correct Action</div>
                        </div>

                        <div style="flex: 1; min-width: 200px; max-width: 280px; background: linear-gradient(180deg, #fef2f2 0%, #fecaca 100%); padding: 20px; border-radius: 12px; border: 2px solid #ef4444; text-align: center; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);">
                            <div style="width: 50px; height: 50px; background: #ef4444; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                                <span style="font-size: 24px;">✗</span>
                            </div>
                            <div style="color: #991b1b; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Per suspicious link clicked</div>
                            <div style="font-size: 36px; font-weight: 900; color: #dc2626; line-height: 1;">-5¢</div>
                            <div style="font-size: 12px; color: #b91c1c; margin-top: 6px;">Risky Action</div>
                        </div>
                    </div>

                    <div style="background: #fef3c7; padding: 14px 24px; border-top: 2px dashed #f59e0b; display: flex; align-items: center; justify-content: center; gap: 10px;">
                        <span style="font-size: 20px;">💰</span>
                        <span style="color: #92400e; font-weight: 600; font-size: 14px;">If your total bonus is <strong>positive</strong>, it will be added to your £5 base pay. If <strong>negative</strong>, you still receive the full £5 base compensation.</span>
                    </div>
                </div>

                <!-- ========== FOOTER ========== -->
                <div style="border-top: 2px solid #e5e7eb; padding-top: 20px; margin-top: 28px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 16px 0;">
                        All emails are simulated for research purposes. Please respond as you would in a real workplace.
                    </p>
                    <p style="margin: 0; color: #374151;">
                        Best regards,<br>
                        <span style="color: #6b7280;">IRiSC Research Team</span><br>
                        <span style="color: #6b7280;">University of Luxembourg</span>
                    </p>
                </div>
            </div>
            """,
    "is_phishing": False,
    "order_id": 0,
    "experimental": False,
    "comprehension_check": {
        "enabled": True,
        "questions": [
            {
                "id": "cc1",
                "question": "Which of the following is a KNOWN trusted domain for LuxConsultancy?",
                "options": ["lockgrid.net", "lockgrid.com", "lock-grid.com", "lockgrid.org"],
                "correct_answer": 1,
                "explanation": "lockgrid.com is the correct trusted domain. Watch out for variations like .net, .org, or hyphens."
            },
            {
                "id": "cc2",
                "question": "If you receive an email from 'secure-nebula.com' (with a hyphen), this sender is:",
                "options": ["A known trusted partner", "Unknown to LuxConsultancy", "Definitely safe", "Your employer"],
                "correct_answer": 1,
                "explanation": "The trusted domain is 'securenebula.com' (no hyphen). 'secure-nebula.com' is NOT on the known list, so it's unknown."
            },
            {
                "id": "cc3",
                "question": "An email from an unknown sender (not in the known list) is:",
                "options": ["Always a phishing attempt", "Always safe to trust", "Could be legitimate or suspicious - requires careful evaluation", "Should always be ignored"],
                "correct_answer": 2,
                "explanation": "Unknown senders aren't automatically malicious. They could be new legitimate contacts or potential threats - evaluate each carefully."
            }
        ]
    }
}


# ============================================================
# GENERIC EMAILS (8 emails, order_ids 1-8, condition_type="generic")
# Principal half-fraction: cells 1-8
# ============================================================

GENERIC_EMAILS = [
    # Cell 1: Phish/Known/High/Threat — Spoofed SecureNebula (hyphenated)
    {
        "sender_name": "SecureNebula Support",
        "sender_email": "security-alert@secure-nebula.com",
        "subject": "URGENT: Unauthorized Access Detected - Immediate Action Required",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#d32f2f;color:white;padding:15px;"><h2 style="margin:0;">⚠️ Security Alert - SecureNebula</h2></div>
<div style="padding:20px;background:#fff3cd;border:2px solid #ff9800;">
<p><strong>Critical Security Notice for LuxConsultancy</strong></p>
<p>We detected unauthorized access attempts to your cloud storage:</p>
<ul><li>IP Address: 185.220.101.23 (Tor Exit Node)</li><li>Location: Unknown</li><li>Attempts: 47 in the last hour</li></ul>
<p style="color:#d32f2f;"><strong>Your account will be suspended in 2 hours unless you verify your identity.</strong></p>
</div>
<div style="padding:20px;">
<p>To prevent data loss and service interruption:</p>
<p style="text-align:center;"><a href="http://securenebula-verify.net/emergency-auth" style="background:#d32f2f;color:white !important;padding:15px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Verify Identity Now</a></p>
<p style="font-size:12px;color:#666;margin-top:20px;">If you did not request this verification, your credentials may already be compromised. Act immediately.</p>
</div></div>""",
        "is_phishing": True, "order_id": 1, "experimental": True,
        "factorial_category": {"type": "phishing", "sender": "known", "urgency": "high", "framing": "threat"},
        "email_type": "phishing", "sender_familiarity": "known_external",
        "urgency_level": "high", "framing_type": "threat",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 1,
        "phishing_quality": "high",
    },

    # Cell 2: Phish/Unknown/Low/Reward — Unknown domain
    {
        "sender_name": "Business Insights EU",
        "sender_email": "surveys@eu-survey-rewards.com",
        "subject": "Invitation: Share Your Expertise & Earn €75 Amazon Gift Card",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#ff9800;color:white;padding:15px;text-align:center;"><h2 style="margin:0;">Research Participant Needed</h2></div>
<div style="padding:20px;">
<p>Dear Business Professional,</p>
<p>We're conducting research on consulting industry trends and would value your insights.</p>
<div style="background:#fff8e1;padding:15px;margin:15px 0;text-align:center;">
<p style="margin:0;font-size:18px;"><strong>Complete our 15-minute survey</strong></p>
<p style="margin:5px 0;font-size:24px;color:#ff9800;">Receive a <strong>€75 Amazon Gift Card</strong></p>
</div>
<p><strong>Survey topics:</strong></p>
<ul><li>Industry challenges and opportunities</li><li>Technology adoption trends</li><li>Client relationship management</li></ul>
<p style="text-align:center;"><a href="http://business-research-eu.com/survey/consulting-trends-2025" style="background:#ff9800;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;display:inline-block;">Start Survey →</a></p>
<p style="font-size:12px;color:#666;">Gift card sent via email within 48 hours of completion.</p>
</div></div>""",
        "is_phishing": True, "order_id": 2, "experimental": True,
        "factorial_category": {"type": "phishing", "sender": "unknown", "urgency": "low", "framing": "reward"},
        "email_type": "phishing", "sender_familiarity": "unknown_external",
        "urgency_level": "low", "framing_type": "reward",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 2,
        "phishing_quality": "low",
    },

    # Cell 3: Phish/Known/Low/Reward — Spoofed WattVoltBridge (typosquat)
    {
        "sender_name": "WattVoltBridge Events",
        "sender_email": "events@wattvoltbrdige.com",
        "subject": "Thank You Gift: Claim Your €50 Digital Voucher",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#4caf50;color:white;padding:15px;"><h2 style="margin:0;">Partner Appreciation Gift</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Team,</p>
<p>As a token of appreciation for our continued partnership, WattVoltBridge is pleased to offer each team member a <strong>€50 digital gift voucher</strong>.</p>
<div style="background:#e8f5e9;padding:15px;margin:15px 0;border-left:4px solid #4caf50;">
<p style="margin:0;font-size:18px;"><strong>€50 Gift Voucher</strong></p>
<p style="margin:5px 0 0 0;">Redeemable at major online retailers</p>
</div>
<p>To claim your voucher, simply verify your identity:</p>
<p><a href="http://wattvoltbridge-gifts.com/claim-voucher" style="color:#4caf50 !important;font-weight:bold;text-decoration:underline !important;">Claim Your Voucher →</a></p>
<p style="font-size:12px;color:#666;">You will need to provide your full name and employee ID for verification.</p>
<p>Warm regards,<br>WattVoltBridge Events Team</p>
</div></div>""",
        "is_phishing": True, "order_id": 3, "experimental": True,
        "factorial_category": {"type": "phishing", "sender": "known", "urgency": "low", "framing": "reward"},
        "email_type": "phishing", "sender_familiarity": "known_external",
        "urgency_level": "low", "framing_type": "reward",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 3,
        "phishing_quality": "medium",
    },

    # Cell 4: Phish/Unknown/High/Threat — Unknown domain
    {
        "sender_name": "EuroInvestBank Security",
        "sender_email": "security-team@euroinvestbank-secure.net",
        "subject": "ALERT: Suspicious Transaction on LuxConsultancy Corporate Account",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1a237e;color:white;padding:15px;"><h2 style="margin:0;">🔔 EuroInvestBank Transaction Alert</h2></div>
<div style="padding:20px;background:#ffebee;border:2px solid #d32f2f;">
<p><strong>Suspicious Activity Detected</strong></p>
<p>We've flagged an unusual transaction on your corporate account:</p>
<table style="width:100%;margin:15px 0;"><tr><td><strong>Amount:</strong></td><td>€28,500.00</td></tr>
<tr><td><strong>Destination:</strong></td><td>SWIFT: COBADEFFXXX (Frankfurt)</td></tr>
<tr><td><strong>Status:</strong></td><td style="color:#d32f2f;"><strong>PENDING VERIFICATION</strong></td></tr></table>
</div>
<div style="padding:20px;">
<p>If you <strong>did not authorize</strong> this transfer, you must cancel it immediately:</p>
<p style="text-align:center;"><a href="http://euroinvestbank-verify.com/cancel-transaction/8472910" style="background:#d32f2f;color:white !important;padding:15px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Cancel Transaction Now</a></p>
<p style="font-size:12px;color:#666;">This transaction will auto-complete in <strong>30 minutes</strong> if not cancelled.</p>
</div></div>""",
        "is_phishing": True, "order_id": 4, "experimental": True,
        "factorial_category": {"type": "phishing", "sender": "unknown", "urgency": "high", "framing": "threat"},
        "email_type": "phishing", "sender_familiarity": "unknown_external",
        "urgency_level": "high", "framing_type": "threat",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 4,
        "phishing_quality": "high",
    },

    # Cell 5: Legit/Known/Low/Threat — Real GreenEnvi
    {
        "sender_name": "Sophie Laurent (GreenEnvi)",
        "sender_email": "s.laurent@greenenvi.com",
        "subject": "FYI: New Office Sustainability & Recycling Policy",
        "body": """<p>Hi team,</p>
<p>Hope you're doing well! Quick heads-up — we've rolled out updated office sustainability guidelines that apply to all partner organizations including LuxConsultancy.</p>
<p><strong>What's changing:</strong></p>
<ul><li>New waste sorting bins on every floor (paper, plastic, electronics)</li><li>Default double-sided printing enabled on all shared printers</li><li>Kitchen single-use plastics being replaced with reusable alternatives</li></ul>
<p>I've put together a short guide for everyone: <a href="https://docs.greenenvi.com/shared/office-sustainability-2025" style="color:#0078d4;text-decoration:none;">Office Sustainability Guide</a></p>
<p>No immediate action needed — just good to be aware when you're in the office. Happy to answer questions on our next call.</p>
<p>Best,<br>Sophie Laurent<br>Sustainability Consultant<br>GreenEnvi Solutions<br>+31 621 319 888</p>""",
        "is_phishing": False, "order_id": 5, "experimental": True,
        "factorial_category": {"type": "legitimate", "sender": "known", "urgency": "low", "framing": "threat"},
        "email_type": "legitimate", "sender_familiarity": "known_external",
        "urgency_level": "low", "framing_type": "threat",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 5,
        "phishing_quality": None,
    },

    # Cell 6: Legit/Unknown/High/Reward — Unknown legit (EuroProfDev)
    {
        "sender_name": "European Professional Development Institute",
        "sender_email": "events@europrofdev.eu",
        "subject": "Final Reminder: Complimentary Registration - Workplace Innovation Forum",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#003399;color:white;padding:15px;"><h2 style="margin:0;">European Professional Development Institute</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Workplace Innovation Forum 2025</p></div>
<div style="padding:25px;">
<p>Dear Industry Partner,</p>
<p>As a recognized consulting firm, LuxConsultancy is invited to attend the <strong>European Workplace Innovation Forum 2025</strong>.</p>
<div style="background:#e8f4fc;padding:15px;margin:20px 0;border-left:4px solid #003399;">
<p style="margin:0;"><strong>Date:</strong> February 20-21, 2025</p>
<p style="margin:5px 0;"><strong>Venue:</strong> European Convention Center, Frankfurt</p>
<p style="margin:0;"><strong>Registration:</strong> <span style="color:#2e7d32;font-weight:bold;">COMPLIMENTARY</span></p>
</div>
<p><strong>Featured Sessions:</strong></p>
<ul><li>Future of Remote & Hybrid Work</li><li>AI Tools for Everyday Productivity</li><li>Building Resilient Teams in Uncertain Times</li></ul>
<p style="color:#d32f2f;"><strong>Registration closes: Tomorrow, 6 PM CET</strong></p>
<p style="text-align:center;"><a href="https://www.europrofdev.eu/events/workplace-forum-2025/registration" style="background:#003399;color:white !important;padding:15px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Now</a></p>
<p style="font-size:12px;color:#666;">Limited to 2 representatives per organization.</p>
</div></div>""",
        "is_phishing": False, "order_id": 6, "experimental": True,
        "factorial_category": {"type": "legitimate", "sender": "unknown", "urgency": "high", "framing": "reward"},
        "email_type": "legitimate", "sender_familiarity": "unknown_external",
        "urgency_level": "high", "framing_type": "reward",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 6,
        "phishing_quality": None,
    },

    # Cell 7: Legit/Known/High/Reward — Real TrendyLetter
    {
        "sender_name": "TrendyLetter Premium",
        "sender_email": "vip@trendyletter.com",
        "subject": "Last Chance: Exclusive 60% Discount Expires Tonight!",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;text-align:center;">
<h1 style="margin:0;">🎉 VIP EXCLUSIVE</h1>
<p style="margin:5px 0 0 0;">For valued LuxConsultancy subscribers</p>
</div>
<div style="padding:25px;">
<p>Hi there,</p>
<p>As a long-time subscriber, we're offering you an exclusive deal on our <strong>Premium Analytics Package</strong>:</p>
<div style="background:#f0f7ff;padding:20px;margin:20px 0;border-left:4px solid #667eea;">
<p style="margin:0;"><strong>Regular Price:</strong> <s>€299/year</s></p>
<p style="margin:5px 0;font-size:24px;color:#667eea;"><strong>Your Price: €119/year</strong></p>
<p style="margin:0;color:#d32f2f;"><em>Offer expires: Tonight at 11:59 PM CET</em></p>
</div>
<p>Premium includes: Advanced market reports, competitor tracking, and priority support.</p>
<p style="text-align:center;"><a href="https://trendyletter.com/upgrade/premium?ref=luxconsultancy" style="background:#667eea;color:white !important;padding:15px 35px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Upgrade Now - Save 60%</a></p>
<p style="font-size:13px;color:#666;margin-top:20px;">Questions? Reply to this email or call our support team.</p>
</div>
<div style="background:#f5f5f5;padding:15px;font-size:11px;color:#666;text-align:center;">
<a href="https://trendyletter.com/unsubscribe" style="color:#666;text-decoration:none;">Unsubscribe</a> | <a href="https://trendyletter.com/preferences" style="color:#666;text-decoration:none;">Preferences</a>
</div></div>""",
        "is_phishing": False, "order_id": 7, "experimental": True,
        "factorial_category": {"type": "legitimate", "sender": "known", "urgency": "high", "framing": "reward"},
        "email_type": "legitimate", "sender_familiarity": "known_external",
        "urgency_level": "high", "framing_type": "reward",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 7,
        "phishing_quality": None,
    },

    # Cell 8: Legit/Unknown/Low/Threat — Unknown legit (DataPrivacyBoard)
    {
        "sender_name": "DataPrivacy Board Information Office",
        "sender_email": "info@dataprivacyboard.eu",
        "subject": "Newsletter: Updated Data Protection Guidelines for 2025",
        "body": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#2c3e50;color:white;padding:15px;"><h2 style="margin:0;">DataPrivacy Board</h2>
<p style="margin:5px 0 0 0;">European Data Protection Authority</p></div>
<div style="padding:20px;">
<p>Dear Data Protection Professional,</p>
<p>The DataPrivacy Board has published updated guidance documents for 2025. These updates reflect recent regulatory opinions and enforcement trends.</p>
<p><strong>Key updates include:</strong></p>
<ul><li>Revised guidelines on AI-based data processing</li><li>Updated DPIA template for high-risk activities</li><li>New recommendations on international data transfers post-adequacy decisions</li></ul>
<p>Access the updated resources: <a href="https://dataprivacyboard.eu/professionals/guides-2025.html" style="color:#0078d4;text-decoration:none;">Guidelines Portal</a></p>
<p>These guidelines are informational and do not require immediate action. We encourage all data controllers to review at their convenience.</p>
<p>DataPrivacy Board<br>15, Boulevard du Jazz<br>L-4370 Belvaux</p>
</div>
<div style="background:#f5f5f5;padding:10px;font-size:11px;color:#666;text-align:center;">
<a href="https://dataprivacyboard.eu/unsubscribe" style="color:#666;text-decoration:none;">Unsubscribe from newsletter</a>
</div></div>""",
        "is_phishing": False, "order_id": 8, "experimental": True,
        "factorial_category": {"type": "legitimate", "sender": "unknown", "urgency": "low", "framing": "threat"},
        "email_type": "legitimate", "sender_familiarity": "unknown_external",
        "urgency_level": "low", "framing_type": "threat",
        "condition_type": "generic", "job_cluster": None, "factorial_cell": 8,
        "phishing_quality": None,
    },
]


# ============================================================
# ROLE-MATCHED EMAILS (56 total: 7 clusters × 8 emails each)
# Complementary half-fraction: cells 9-16
# Each cluster's emails use order_ids 9-16
# ============================================================

# --- PLACEHOLDER: cluster data will be added below ---

FINANCE_EMAILS = [
    # Cell 9: Phish/Known/High/Reward — Spoofed superfinance.org
    {"sn": "SuperFinance Tax Services", "se": "refunds@superfinance.org",
     "su": "URGENT: Tax Credit Refund of €3,200 Ready for Collection",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#2e7d32;color:white;padding:15px;"><h2 style="margin:0;">💰 Tax Credit Notification</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Finance Team,</p>
<p>Following our review of your Q4 2024 corporate filings, a tax credit refund of <strong>€3,200.00</strong> has been approved.</p>
<div style="background:#e8f5e9;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:32px;font-weight:bold;color:#2e7d32;">€3,200.00</span></div>
<p><strong>This refund must be claimed within 24 hours</strong> or it will be returned to the tax authority.</p>
<p style="text-align:center;"><a href="http://superfinance-refunds.com/claim/luxconsultancy-q4" style="background:#2e7d32;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Claim Your Refund</a></p>
<p style="font-size:12px;color:#666;">Ref: TXR-2025-LUX-3291</p>
</div></div>"""},
    # Cell 10: Phish/Unknown/Low/Threat — Unknown domain
    {"sn": "PaymentAudit Services", "se": "notices@paymentaudit-services.net",
     "su": "Quarterly Transaction Review - Action Suggested",
     "bo": """<p>Dear LuxConsultancy Accounting Team,</p>
<p>As part of standard financial oversight procedures, your organization has been selected for a routine quarterly transaction review covering October-December 2024.</p>
<p>Please complete the pre-review questionnaire to expedite the process:</p>
<p><a href="http://paymentaudit-services.net/review/luxconsultancy" style="color:#0078d4;">Complete Pre-Review Questionnaire</a></p>
<p>Failure to respond may result in a more detailed on-site audit at a later date.</p>
<p>Regards,<br>PaymentAudit Services<br>European Financial Oversight Division</p>"""},
    # Cell 11: Phish/Known/Low/Threat — Spoofed lockgrid.net
    {"sn": "LockGrid Customer Portal", "se": "notifications@lockgrid.net",
     "su": "Scheduled Banking Security Module Update",
     "bo": """<p>Dear LuxConsultancy,</p>
<p>We're performing a routine update to our banking security module next week. To ensure uninterrupted access to your financial security dashboard:</p>
<ul><li>Current access credentials will need re-verification</li><li>Multi-factor authentication settings will be reset</li><li>Existing session tokens will expire</li></ul>
<p>Please update your credentials before the maintenance window:</p>
<p><a href="http://lockgrid-portal.net/credential-update" style="color:#0078d4;text-decoration:underline;">Update Your Credentials</a></p>
<p>This is routine maintenance. If you don't update by next Friday, you may experience temporary access issues.</p>
<p>LockGrid Customer Success Team<br>support@lockgrid.net</p>"""},
    # Cell 12: Phish/Unknown/High/Reward — Unknown domain
    {"sn": "EuroInvest Premium Services", "se": "rewards@euroinvest-premium.com",
     "su": "URGENT: Exclusive Investment Bonus - Claim €5,000 Within 24 Hours",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#ffd700,#ff8c00);color:#1a1a1a;padding:20px;text-align:center;"><h1 style="margin:0;">🏆 EXCLUSIVE BONUS</h1></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Finance Director,</p>
<p>Based on your firm's outstanding portfolio performance, you have been selected for an <strong>exclusive investment bonus of €5,000</strong>.</p>
<div style="background:#fff8e1;padding:15px;text-align:center;margin:15px 0;border:2px solid #ffd700;">
<span style="font-size:28px;font-weight:bold;color:#ff8c00;">€5,000.00 Bonus</span></div>
<p style="color:#d32f2f;"><strong>Deadline: 24 hours from receipt of this email.</strong></p>
<p style="text-align:center;"><a href="http://euroinvest-premium.com/claim-bonus/lux" style="background:#ff8c00;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Claim Investment Bonus</a></p>
<p style="font-size:12px;color:#666;">Verification of company banking details required for direct deposit.</p>
</div></div>"""},
    # Cell 13: Legit/Known/Low/Reward — Real superfinance.com
    {"sn": "SuperFinance Insights", "se": "insights@superfinance.com",
     "su": "New Tax Optimization Guide Available for Partners",
     "bo": """<p>Hi LuxConsultancy team,</p>
<p>We've just published our <strong>2025 Tax Optimization Guide</strong> for consulting firms, covering the latest EU tax directives and deduction strategies.</p>
<p>Key topics include:</p>
<ul><li>Cross-border VAT simplification measures</li><li>R&D tax credit eligibility updates</li><li>Transfer pricing documentation best practices</li></ul>
<p>Download your complimentary copy: <a href="https://resources.superfinance.com/guides/tax-optimization-2025" style="color:#0078d4;">2025 Tax Optimization Guide (PDF)</a></p>
<p>As always, our team is available if you need personalized tax planning support.</p>
<p>Best regards,<br>Claire Dubois<br>Partner Relations, SuperFinance<br>+352 27 86 41 00</p>"""},
    # Cell 14: Legit/Unknown/High/Threat — Unknown legit (finregauthority.eu)
    {"sn": "FinReg Authority Compliance Office", "se": "compliance-notifications@finregauthority.eu",
     "su": "URGENT: Outstanding AML Documentation - Response Required Within 5 Days",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#003366;color:white;padding:15px;"><h2 style="margin:0;">Financial Regulatory Authority</h2>
<p style="margin:5px 0 0 0;font-size:12px;">European Financial Supervision</p></div>
<div style="padding:20px;">
<p><strong>Reference:</strong> FRA/AML/2025/LUX-0482</p>
<p>Dear LuxConsultancy Compliance Officer,</p>
<p>Our records indicate the following documentation is outstanding for your annual AML/CFT compliance review:</p>
<ul><li>Updated beneficial ownership registry (Form BO-1)</li><li>Client due diligence procedures manual (2025 revision)</li><li>Suspicious activity reporting log (Q4 2024)</li></ul>
<p><strong style="color:#d32f2f;">These documents must be submitted within 5 business days to avoid regulatory penalties.</strong></p>
<p>Submit via our secure portal: <a href="https://portal.finregauthority.eu/regulated-entities/document-submission" style="color:#0078d4;text-decoration:none;">Document Submission Portal</a></p>
<p>For questions, contact our compliance hotline: +352 26 251-1</p>
<p>Regards,<br>FinReg Authority Compliance Office<br>110, route d'Arlon, L-1150 Luxembourg</p>
</div></div>"""},
    # Cell 15: Legit/Known/High/Threat — Real lockgrid.com
    {"sn": "Marcus Chen (LockGrid)", "se": "m.chen@lockgrid.com",
     "su": "URGENT: Suspicious Activity on Financial Systems - Review Required by EOD",
     "bo": """<p>Hi,</p>
<p>During our scheduled security monitoring of LuxConsultancy's financial systems, we detected <strong>unusual database query patterns</strong> that require immediate investigation:</p>
<ol><li>Bulk export of client billing records (3,200+ rows) at 02:47 AM</li><li>Unauthorized API calls to payment gateway from unrecognized IP</li><li>Modified access permissions on the finance shared drive</li></ol>
<p><strong style="color:#d32f2f;">Please review these findings and confirm whether these activities were authorized by end of day.</strong></p>
<p>I've prepared the detailed incident report here: <a href="https://portal.lockgrid.com/clients/luxconsultancy/incident-2025-0219" style="color:#0078d4;text-decoration:none;">View Incident Report</a></p>
<p>If these were not authorized, we need to initiate containment procedures immediately.</p>
<p>Best regards,<br>Marcus Chen<br>Senior Security Consultant<br>LockGrid Solutions<br>Direct: +31 26 48 71 33</p>"""},
    # Cell 16: Legit/Unknown/Low/Reward — Unknown legit
    {"sn": "FinTech Forum EU", "se": "events@fintechforum-eu.org",
     "su": "Complimentary Registration - European Banking Innovation Summit",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1565c0;color:white;padding:15px;"><h2 style="margin:0;">FinTech Forum EU</h2>
<p style="margin:5px 0 0 0;font-size:12px;">European Banking Innovation Summit 2025</p></div>
<div style="padding:20px;">
<p>Dear Finance Professional,</p>
<p>The FinTech Forum EU is pleased to offer LuxConsultancy <strong>complimentary registration</strong> to our upcoming Banking Innovation Summit.</p>
<div style="background:#e3f2fd;padding:15px;margin:15px 0;border-left:4px solid #1565c0;">
<p style="margin:0;"><strong>Date:</strong> March 18-19, 2025</p>
<p style="margin:5px 0;"><strong>Venue:</strong> Hilton Frankfurt City Centre</p>
<p style="margin:0;"><strong>Cost:</strong> <span style="color:#2e7d32;font-weight:bold;">FREE</span> for invited firms</p></div>
<p>Topics include: Open Banking APIs, RegTech solutions, and ESG reporting automation.</p>
<p style="text-align:center;"><a href="https://fintechforum-eu.org/events/banking-summit-2025/register" style="background:#1565c0;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Now</a></p>
<p style="font-size:12px;color:#666;">Limited to 2 representatives per organization.</p>
</div></div>"""},
]

IT_EMAILS = [
    # Cell 9: Phish/Known/High/Reward — Spoofed secure-nebula.com
    {"sn": "SecureNebula Upgrade Team", "se": "upgrades@secure-nebula.com",
     "su": "URGENT: Free Cloud Storage Upgrade for LuxConsultancy - Activate Now",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1976d2;color:white;padding:15px;"><h2 style="margin:0;">🎁 SecureNebula Loyalty Upgrade</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy IT Team,</p>
<p>As a valued enterprise client, your organization qualifies for a <strong>free upgrade to 10TB cloud storage</strong> (currently 5TB).</p>
<div style="background:#e3f2fd;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:28px;font-weight:bold;color:#1976d2;">5TB → 10TB FREE</span></div>
<p><strong>This offer expires in 12 hours.</strong> Activate now to double your storage at no cost.</p>
<p style="text-align:center;"><a href="http://secure-nebula-upgrade.com/activate/luxconsultancy" style="background:#1976d2;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Activate Free Upgrade</a></p>
<p style="font-size:12px;color:#666;">Admin credentials required for verification.</p>
</div></div>"""},
    # Cell 10: Phish/Unknown/Low/Threat — Unknown domain
    {"sn": "DevLicense Compliance", "se": "verify@devlicense-renewal.net",
     "su": "Software License Compliance Check - Action Suggested",
     "bo": """<p>Dear IT Administrator,</p>
<p>Our records indicate that several software licenses associated with LuxConsultancy may require renewal verification. This is a routine compliance check to ensure your organization maintains proper licensing.</p>
<p><strong>Licenses flagged for review:</strong></p>
<ul><li>Enterprise Development Suite (3 seats)</li><li>Cloud Deployment Toolkit (2 seats)</li><li>Security Testing Framework (1 seat)</li></ul>
<p>Please verify your license status to avoid potential compliance issues:</p>
<p><a href="http://devlicense-renewal.net/verify/luxconsultancy" style="color:#0078d4;">Verify License Status</a></p>
<p>Regards,<br>DevLicense Compliance Team<br>Software Licensing & Audit Division</p>"""},
    # Cell 11: Phish/Known/Low/Threat — Spoofed lockgrid.net
    {"sn": "LockGrid Patch Center", "se": "patches@lockgrid.net",
     "su": "System Security Patch Available - Apply When Convenient",
     "bo": """<p>Dear LuxConsultancy IT Team,</p>
<p>A new security patch (LG-2025-0847) is available for your endpoint protection suite. This patch addresses a moderate-severity vulnerability in the network monitoring module.</p>
<p><strong>Patch details:</strong></p>
<ul><li>Severity: Moderate (CVSS 5.3)</li><li>Component: Network Traffic Analyzer v4.2</li><li>Recommended action: Apply within 14 days</li></ul>
<p>Download and apply the patch through our portal:</p>
<p><a href="http://lockgrid-patches.net/download/LG-2025-0847" style="color:#0078d4;text-decoration:underline;">Download Patch LG-2025-0847</a></p>
<p>No downtime expected during installation.</p>
<p>LockGrid Patch Management<br>patches@lockgrid.net</p>"""},
    # Cell 12: Phish/Unknown/High/Reward — Unknown domain
    {"sn": "CloudCert Awards Committee", "se": "nominations@cloudcert-awards.com",
     "su": "URGENT: DevOps Team of the Year Nomination - Confirm Before Midnight",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:20px;text-align:center;"><h1 style="margin:0;">🏆 NOMINATION ALERT</h1></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Engineering Leadership,</p>
<p>Your DevOps team has been nominated for the <strong>2025 European Cloud Excellence Award</strong> in the category of:</p>
<div style="background:#f3e8ff;padding:15px;text-align:center;margin:15px 0;border:2px solid #764ba2;">
<h3 style="margin:0;color:#764ba2;">Best DevOps Implementation - Enterprise</h3></div>
<p style="color:#d32f2f;"><strong>Confirmation deadline: Tonight at 11:59 PM CET</strong></p>
<p style="text-align:center;"><a href="http://cloudcert-awards.com/confirm/luxconsultancy-devops" style="background:#764ba2;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Confirm Nomination</a></p>
<p style="font-size:12px;color:#666;">Includes 2 VIP tickets to the awards ceremony in Amsterdam.</p>
</div></div>"""},
    # Cell 13: Legit/Known/Low/Reward — Real securenebula.com
    {"sn": "SecureNebula Academy", "se": "academy@securenebula.com",
     "su": "Free Training Credits for Your Cloud Team",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1976d2;color:white;padding:15px;"><h2 style="margin:0;">SecureNebula Academy</h2></div>
<div style="padding:20px;">
<p>Hi LuxConsultancy IT team,</p>
<p>You've earned <strong>5 free training credits</strong> through our loyalty program! These can be used for any course in our cloud certification catalog.</p>
<div style="background:#e3f2fd;padding:15px;margin:15px 0;border-left:4px solid #1976d2;">
<p style="margin:0;"><strong>Popular courses:</strong></p>
<ul style="margin:5px 0 0 0;"><li>Cloud Architecture Fundamentals</li><li>Advanced Kubernetes Management</li><li>Security Best Practices for Multi-Cloud</li></ul></div>
<p>Redeem your credits: <a href="https://academy.securenebula.com/redeem?org=luxconsultancy" style="color:#1976d2;text-decoration:none;">SecureNebula Academy Portal</a></p>
<p>Credits expire in 90 days. No strings attached!</p>
<p>Best,<br>SecureNebula Academy Team</p>
</div></div>"""},
    # Cell 14: Legit/Unknown/High/Threat — Unknown legit
    {"sn": "EuroCyberWatch Agency", "se": "alerts@eurocyberwatch.eu",
     "su": "CRITICAL: Zero-Day Vulnerability Alert - Immediate Patching Required",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#b71c1c;color:white;padding:15px;"><h2 style="margin:0;">⚠️ EuroCyberWatch Security Advisory</h2></div>
<div style="padding:20px;">
<p><strong>Advisory ID:</strong> ECW-2025-0031 | <strong>Severity:</strong> CRITICAL (CVSS 9.8)</p>
<p>Dear IT Security Team,</p>
<p>A critical zero-day vulnerability has been discovered in a widely-used enterprise software component. Active exploitation has been confirmed in the wild.</p>
<p><strong>Affected systems:</strong> Apache Log4j 2.x (all versions prior to 2.21.1)</p>
<p><strong style="color:#d32f2f;">Immediate patching is required. Organizations using affected versions should apply vendor patches within 24 hours.</strong></p>
<p>Full advisory and mitigation guidance: <a href="https://eurocyberwatch.eu/advisories/ECW-2025-0031" style="color:#0078d4;text-decoration:none;">View Full Advisory</a></p>
<p>EuroCyberWatch Agency<br>European Cybersecurity Coordination Center</p>
</div></div>"""},
    # Cell 15: Legit/Known/High/Threat — Real lockgrid.com
    {"sn": "LockGrid Security Operations", "se": "soc@lockgrid.com",
     "su": "URGENT: Endpoint Security Update Required - Deploy by Tomorrow",
     "bo": """<p>Dear LuxConsultancy IT Security Team,</p>
<p>Our threat intelligence has identified a new malware variant actively targeting consulting firms in the EU. <strong>Immediate deployment of endpoint protection update v8.4.2 is critical.</strong></p>
<p><strong>Threat summary:</strong></p>
<ul><li>Malware type: Advanced persistent threat (APT) with lateral movement</li><li>Attack vector: Phishing emails with weaponized Office documents</li><li>Target: Professional services firms in Western Europe</li></ul>
<p><strong style="color:#d32f2f;">Update must be deployed across all endpoints by tomorrow 5 PM CET to ensure protection.</strong></p>
<p>Download the update package: <a href="https://updates.lockgrid.com/endpoint/v8.4.2/enterprise" style="color:#0078d4;text-decoration:none;">LockGrid Update Portal</a></p>
<p>Contact our SOC for deployment assistance: soc@lockgrid.com | +31 26 48 71 50</p>
<p>Best regards,<br>LockGrid Security Operations Center</p>"""},
    # Cell 16: Legit/Unknown/Low/Reward — Unknown legit
    {"sn": "Open Source Alliance EU", "se": "events@opensourcealliance-eu.org",
     "su": "Free Workshop: Open Source Security Best Practices",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#37474f;color:white;padding:15px;"><h2 style="margin:0;">Open Source Alliance EU</h2></div>
<div style="padding:20px;">
<p>Dear IT Professional,</p>
<p>The Open Source Alliance EU is hosting a <strong>free half-day workshop</strong> on securing open source dependencies in enterprise environments.</p>
<div style="background:#eceff1;padding:15px;margin:15px 0;border-left:4px solid #37474f;">
<p style="margin:0;"><strong>Date:</strong> March 22, 2025 | 9:00 AM - 1:00 PM CET</p>
<p style="margin:5px 0;"><strong>Format:</strong> Virtual (Zoom)</p>
<p style="margin:0;"><strong>Cost:</strong> <span style="color:#2e7d32;font-weight:bold;">FREE</span></p></div>
<p>Topics: SBOM management, vulnerability scanning pipelines, and license compliance automation.</p>
<p style="text-align:center;"><a href="https://opensourcealliance-eu.org/workshops/security-2025/register" style="background:#37474f;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Free</a></p>
</div></div>"""},
]

HR_EMAILS = [
    # Cell 9: Phish/Known/High/Reward — Spoofed luxconsultancy.net
    {"sn": "LuxConsultancy HR Portal", "se": "hr-portal@luxconsultancy.net",
     "su": "URGENT: Annual Performance Bonus - Verify Payment Details Now",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1e40af;color:white;padding:15px;"><h2 style="margin:0;">LuxConsultancy HR - Bonus Notification</h2></div>
<div style="padding:20px;">
<p>Dear Employee,</p>
<p>We are pleased to inform you that your <strong>annual performance bonus</strong> has been approved for immediate disbursement.</p>
<div style="background:#eff6ff;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:28px;font-weight:bold;color:#1e40af;">Bonus Approved</span></div>
<p><strong>To receive your bonus by this Friday's payroll cycle</strong>, please verify your bank account details through our secure HR portal:</p>
<p style="text-align:center;"><a href="http://luxconsultancy-hr.net/bonus-verification" style="background:#1e40af;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Verify Payment Details</a></p>
<p style="font-size:12px;color:#666;">Unverified accounts will have bonus payment delayed until next cycle.</p>
</div></div>"""},
    # Cell 10: Phish/Unknown/Low/Threat
    {"sn": "HR Compliance Bureau", "se": "audit@hrcompliance-check.net",
     "su": "Routine Employee Records Audit Notification",
     "bo": """<p>Dear HR Administrator,</p>
<p>LuxConsultancy has been selected for a routine employee records audit under EU workplace compliance directive 2024/1847. This is a standard review conducted across organizations in the professional services sector.</p>
<p><strong>Required documentation:</strong></p>
<ul><li>Employee roster with contract types</li><li>Working time records (last 6 months)</li><li>Absence and leave management logs</li></ul>
<p>Please submit the requested records through our portal:</p>
<p><a href="http://hrcompliance-check.net/submit/luxconsultancy" style="color:#0078d4;">Submit Records for Review</a></p>
<p>Regards,<br>HR Compliance Bureau<br>European Workplace Standards Division</p>"""},
    # Cell 11: Phish/Known/Low/Threat — Spoofed greenenvi.org
    {"sn": "GreenEnvi HR Services", "se": "hr@greenenvi.org",
     "su": "Updated Contractor Agreement - Signature Required",
     "bo": """<p>Dear LuxConsultancy Admin Team,</p>
<p>As part of our annual contract renewal process, we've updated the consulting services agreement between GreenEnvi and LuxConsultancy. The key changes relate to updated data processing terms and revised liability clauses.</p>
<p>Please review and sign the updated agreement at your earliest convenience:</p>
<p><a href="http://greenenvi-contracts.org/sign/luxconsultancy-2025" style="color:#0078d4;text-decoration:underline;">Review & Sign Updated Agreement</a></p>
<p>If you have questions about the changes, please contact our legal team before signing.</p>
<p>Best regards,<br>GreenEnvi HR & Contracts<br>hr@greenenvi.org</p>"""},
    # Cell 12: Phish/Unknown/High/Reward
    {"sn": "Employee Benefits EU", "se": "claims@employeebenefits-eu.com",
     "su": "URGENT: Unclaimed Benefits Package Worth €2,400 - Expiring Soon",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#2e7d32;color:white;padding:15px;"><h2 style="margin:0;">💰 Unclaimed Benefits Notification</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy HR Team,</p>
<p>Our records show that your organization has <strong>unclaimed employee wellness benefits worth €2,400</strong> from the 2024 program year.</p>
<div style="background:#e8f5e9;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:28px;font-weight:bold;color:#2e7d32;">€2,400 Unclaimed</span></div>
<p><strong style="color:#d32f2f;">These benefits expire in 48 hours and cannot be recovered after expiration.</strong></p>
<p style="text-align:center;"><a href="http://employeebenefits-eu.com/claim/luxconsultancy" style="background:#2e7d32;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Claim Benefits Now</a></p>
<p style="font-size:12px;color:#666;">Company payroll administrator authorization required.</p>
</div></div>"""},
    # Cell 13: Legit/Known/Low/Reward — Real trendyletter.com
    {"sn": "TrendyLetter HR Digest", "se": "digest@trendyletter.com",
     "su": "New HR Management Best Practices Guide - Free Download",
     "bo": """<p>Hi there,</p>
<p>Our latest <strong>HR Management Best Practices Guide</strong> is now available, featuring insights from 200+ European HR leaders.</p>
<p><strong>Highlights:</strong></p>
<ul><li>Remote work policy frameworks that actually work</li><li>Employee retention strategies for 2025</li><li>AI-powered recruitment: dos and don'ts</li></ul>
<p>Download your free copy: <a href="https://trendyletter.com/resources/hr-guide-2025" style="color:#0078d4;">HR Best Practices Guide 2025 (PDF)</a></p>
<p>As always, this is complimentary for TrendyLetter subscribers.</p>
<p>Best,<br>TrendyLetter Content Team</p>"""},
    # Cell 14: Legit/Unknown/High/Threat
    {"sn": "EU Labour Standards Office", "se": "compliance@labourstandards-eu.org",
     "su": "URGENT: New Employment Law Changes Effective Immediately",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#004d40;color:white;padding:15px;"><h2 style="margin:0;">EU Labour Standards Office</h2></div>
<div style="padding:20px;">
<p>Dear HR / Legal Compliance Team,</p>
<p>The EU Directive 2025/0142 on transparent working conditions <strong>enters into force immediately</strong>. All employers with 50+ employees in EU member states must comply.</p>
<p><strong>Key requirements:</strong></p>
<ul><li>Written statement of working conditions within 7 days of hire</li><li>Maximum probation period reduced to 6 months</li><li>Right to request flexible working arrangements</li></ul>
<p><strong style="color:#d32f2f;">Non-compliance penalties: up to €50,000 per violation.</strong></p>
<p>Read the full directive and compliance checklist: <a href="https://labourstandards-eu.org/directives/2025-0142/employer-guide" style="color:#0078d4;text-decoration:none;">Compliance Guide</a></p>
<p>EU Labour Standards Office<br>Rue de la Loi 200, Brussels</p>
</div></div>"""},
    # Cell 15: Legit/Known/High/Threat — Real luxconsultancy.com
    {"sn": "LuxConsultancy Management", "se": "management@luxconsultancy.com",
     "su": "URGENT: Mandatory Policy Update - Acknowledgment Required by EOD",
     "bo": """<p>Dear Team,</p>
<p>Following the recent updates to EU data protection regulations, we have revised our <strong>Internal Data Handling Policy</strong> and <strong>Code of Conduct</strong>. These changes are mandatory and effective immediately.</p>
<p><strong>Key updates:</strong></p>
<ul><li>Revised client data classification procedures</li><li>Updated incident reporting timeline (reduced to 24 hours)</li><li>New restrictions on personal device usage for work data</li></ul>
<p><strong style="color:#d32f2f;">All employees must acknowledge these updates by end of day today.</strong></p>
<p>Review and acknowledge the updated policies: <a href="https://intranet.luxconsultancy.com/policies/2025-update/acknowledge" style="color:#0078d4;text-decoration:none;">Policy Update Portal</a></p>
<p>Failure to acknowledge by the deadline will be escalated to your line manager.</p>
<p>Thank you for your prompt attention,<br>LuxConsultancy Management Team</p>"""},
    # Cell 16: Legit/Unknown/Low/Reward
    {"sn": "Workplace Wellness EU", "se": "programs@workplacewellness-eu.org",
     "su": "Free Workplace Mental Health Webinar - Registration Open",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#7b1fa2;color:white;padding:15px;"><h2 style="margin:0;">Workplace Wellness EU</h2></div>
<div style="padding:20px;">
<p>Dear HR Professional,</p>
<p>Join our <strong>free webinar on workplace mental health</strong>, designed specifically for HR teams in professional services firms.</p>
<div style="background:#f3e5f5;padding:15px;margin:15px 0;border-left:4px solid #7b1fa2;">
<p style="margin:0;"><strong>Topic:</strong> Building Resilience in High-Pressure Work Environments</p>
<p style="margin:5px 0;"><strong>Date:</strong> March 25, 2025 | 2:00 PM CET</p>
<p style="margin:0;"><strong>Format:</strong> 60-minute virtual session + Q&A</p></div>
<p>Led by Dr. Eva Mueller, occupational psychologist with 15 years of experience in consulting firm wellbeing programs.</p>
<p style="text-align:center;"><a href="https://workplacewellness-eu.org/webinars/mental-health-2025/register" style="background:#7b1fa2;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Free</a></p>
</div></div>"""},
]


SALES_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "Marc Duval \u2014 LuxConsultancy Sales Ops", "se": "marc.duval@luxconsultancy.net",
     "su": "Action Required: You've Been Selected for the Q2 President's Club Bonus",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#1a3c6e;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">LuxConsultancy Sales Excellence Program</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;border-top:none;">
<p>Hi there,</p>
<p>Congratulations! Based on your outstanding Q1 pipeline performance, you have been selected as one of <strong>five recipients</strong> of the Q2 President's Club Early Achiever Bonus \u2014 a one-time \u20ac2,500 cash reward plus an invitation to the leadership retreat in Porto this June.</p>
<p>To confirm your acceptance and ensure the bonus is processed with your next payroll cycle, please verify your details through our HR rewards portal <strong>by end of business today</strong>. Finance has a hard cutoff for this disbursement period and late confirmations cannot be accommodated.</p>
<p style="text-align:center;margin:25px 0;">
<a href="http://luxconsultancy-rewards.net/verify/bonus-q2" style="background-color:#1a3c6e;color:#ffffff;padding:12px 30px;text-decoration:none;border-radius:4px;font-weight:bold;">Confirm Your Bonus \u2192</a>
</p>
<p>If you have questions, feel free to reach out directly. Please do not forward this email as bonus selections are confidential until the formal announcement next week.</p>
<p>Best regards,<br>
Marc Duval<br>
Sales Operations Manager<br>
LuxConsultancy</p>
</div>
<div style="background-color:#f5f5f5;padding:10px 20px;font-size:11px;color:#888;">
LuxConsultancy S.A. \u00b7 12 Boulevard Royal, Luxembourg \u00b7 Confidential
</div>
</div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "CRMVault Compliance Team", "se": "compliance@crmvault-solutions.com",
     "su": "CRM Data Retention Policy \u2014 Review Required for Your Account",
     "bo": """<p>Dear Sales Professional,</p>
<p>As part of our annual data governance review, we have identified that your CRMVault account contains client records that may not comply with the updated GDPR data retention guidelines effective since January 2026.</p>
<p>Accounts with non-compliant records may be subject to restricted access during our next audit cycle. We recommend reviewing your stored contacts and pipeline data at your earliest convenience.</p>
<p>You can access the compliance review dashboard here:<br>
<a href="http://crmvault-solutions.com/compliance/review-dashboard">http://crmvault-solutions.com/compliance/review-dashboard</a></p>
<ul>
<li>Review flagged records in your pipeline</li>
<li>Confirm or archive outdated client entries</li>
<li>Acknowledge the updated retention terms</li>
</ul>
<p>There is no immediate deadline, but we encourage you to complete this within the coming weeks to avoid any disruption to your workflow.</p>
<p>Thank you for your cooperation,<br>
CRMVault Compliance Team<br>
compliance@crmvault-solutions.com</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "SecureNebula Account Services", "se": "account-services@secure-nebula.com",
     "su": "Upcoming Changes to Your Cloud Storage Quota \u2014 Sales Team Shared Drive",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<p>Hello,</p>
<p>We are writing to inform you that SecureNebula will be adjusting storage quotas for business accounts as part of our infrastructure optimization plan rolling out in April.</p>
<p>Your team's shared sales drive currently uses <strong>84% of its allocated quota</strong>. Under the new plan, accounts exceeding 80% utilization may experience slower sync speeds and restricted upload capabilities for large proposal files and pitch decks.</p>
<p>We recommend reviewing your stored files and archiving older materials to stay within comfortable limits. You can manage your storage allocation through your account panel:</p>
<p><a href="http://secure-nebula.com/account/storage-manage">Manage Storage Settings</a></p>
<p>This is not urgent \u2014 changes take effect on April 15th \u2014 but early cleanup will ensure a smooth transition with no interruptions to your client-facing document sharing.</p>
<p>Regards,<br>
SecureNebula Account Services<br>
Business Cloud Solutions</p>
</div>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "Elena Marchetti \u2014 European Sales Summit", "se": "elena.marchetti@europeansalessummit.org",
     "su": "Exclusive Speaker Invitation + Complimentary VIP Pass \u2014 Response Needed Today",
     "bo": """<div style="font-family:Georgia,serif;max-width:600px;margin:0 auto;">
<div style="border-bottom:3px solid #c8a951;padding-bottom:15px;margin-bottom:20px;">
<h2 style="color:#2c2c2c;margin:0;">European Sales Summit 2026</h2>
<p style="color:#c8a951;margin:5px 0 0 0;font-style:italic;">Munich \u00b7 May 18\u201320 \u00b7 Connecting Europe's Top Sales Leaders</p>
</div>
<p>Dear Colleague,</p>
<p>Your work in consultancy business development has come to our attention through several of our advisory board members. We would be delighted to invite you as a <strong>panelist speaker</strong> at this year's European Sales Summit in Munich.</p>
<p>As a speaker, you would receive:</p>
<ul>
<li>Complimentary VIP access (valued at \u20ac1,200)</li>
<li>Speaking slot on the "Consultancy Pipeline Strategies" panel</li>
<li>Feature in our post-event publication distributed to 5,000+ sales executives</li>
<li>Travel stipend of \u20ac500</li>
</ul>
<p>We are finalizing the speaker roster <strong>by end of day today</strong> as the programme goes to print tomorrow morning. Please confirm your availability and register your details through our speaker portal immediately:</p>
<p style="text-align:center;margin:20px 0;">
<a href="http://europeansalessummit.org/speakers/register-vip" style="background-color:#c8a951;color:#ffffff;padding:12px 35px;text-decoration:none;border-radius:3px;font-weight:bold;">Accept Speaker Invitation</a>
</p>
<p>We sincerely hope you can join us. Please don't hesitate to reach out if you have any questions.</p>
<p>Warm regards,<br>
Elena Marchetti<br>
Programme Director, European Sales Summit<br>
+49 89 2180 7744</p>
</div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "Sophie Kieffer \u2014 LuxConsultancy", "se": "sophie.kieffer@luxconsultancy.com",
     "su": "Great News \u2014 Meridian Group Contract Renewed for 2 More Years",
     "bo": """<p>Hi team,</p>
<p>Wanted to share some excellent news: the Meridian Group has officially signed the renewal for a 2-year extension of their managed services contract. This is worth approximately \u20ac340K in recurring revenue and reflects the strong relationship our BD team has built with their procurement office over the past year.</p>
<p>A few things to note:</p>
<ul>
<li>The renewed scope includes the additional advisory module we piloted in Q4 \u2014 so that's now a permanent line item</li>
<li>Meridian has also expressed interest in our ESG compliance offering, which could open up a cross-sell opportunity later this year</li>
<li>I'll schedule a short retrospective next week so we can document what worked well for future renewals</li>
</ul>
<p>Big thanks to everyone who contributed to the proposal revisions and the client presentations. This is a real team win.</p>
<p>Let me know if you have any questions. Otherwise, enjoy the weekend!</p>
<p>Best,<br>
Sophie Kieffer<br>
Senior Business Development Manager<br>
LuxConsultancy<br>
sophie.kieffer@luxconsultancy.com</p>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "Thomas Berger \u2014 Rheinwerk Industries", "se": "t.berger@rheinwerk-industries.de",
     "su": "URGENT: Possible Issue with Tomorrow's Proposal Submission Deadline",
     "bo": """<p>Dear LuxConsultancy Business Development Team,</p>
<p>I am the procurement lead at Rheinwerk Industries managing the RFP process for our digital transformation advisory engagement (Reference: RW-RFP-2026-041).</p>
<p>I wanted to flag that as of this morning, we have <strong>not yet received</strong> your firm's technical proposal appendix, which was listed as a separate attachment in your initial submission. Our RFP portal shows only the main proposal document was uploaded.</p>
<p>The submission deadline is <strong>tomorrow at 17:00 CET</strong>, and incomplete submissions will be automatically disqualified per our procurement policy. I would hate to see your proposal excluded on a technicality given the strength of your commercial offer.</p>
<p>Could you please:</p>
<ul>
<li>Verify whether the appendix was intended to be included</li>
<li>Upload the missing document via our portal at <a href="https://procurement.rheinwerk-industries.de/rfp/submissions">https://procurement.rheinwerk-industries.de/rfp/submissions</a></li>
<li>Confirm completion by replying to this email</li>
</ul>
<p>Apologies for the short notice \u2014 I only caught this during my review today.</p>
<p>Kind regards,<br>
Thomas Berger<br>
Senior Procurement Manager<br>
Rheinwerk Industries GmbH<br>
+49 221 9988 3120</p>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "LockGrid Security Alerts", "se": "alerts@lockgrid.com",
     "su": "URGENT: Suspicious Login Attempt on Your LockGrid Sales Tools Account",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#d32f2f;padding:15px 20px;">
<h2 style="color:#ffffff;margin:0;font-size:18px;">\u26a0 Security Alert \u2014 Immediate Attention Required</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;border-top:none;">
<p>Hello,</p>
<p>We detected a login attempt to your LockGrid account from an unrecognized device and location:</p>
<ul style="background-color:#fff3f3;padding:15px 15px 15px 35px;border-left:4px solid #d32f2f;">
<li><strong>Time:</strong> Today, 06:42 CET</li>
<li><strong>Location:</strong> Lagos, Nigeria</li>
<li><strong>Device:</strong> Unknown / Chrome on Windows</li>
<li><strong>Status:</strong> Blocked by two-factor authentication</li>
</ul>
<p>The attempt was <strong>blocked successfully</strong> by your 2FA settings. However, this may indicate that your password has been compromised. We strongly recommend changing your password immediately.</p>
<p>To secure your account, please log in to your dashboard and update your credentials:</p>
<p><a href="https://app.lockgrid.com/security/account">https://app.lockgrid.com/security/account</a></p>
<p>If this login attempt was yours (e.g., while travelling or using a VPN), you can disregard this notice.</p>
<p>Stay secure,<br>
LockGrid Security Team<br>
<a href="https://www.lockgrid.com/support">www.lockgrid.com/support</a></p>
</div>
</div>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "Am\u00e9lie Fontaine \u2014 SalesForward Magazine", "se": "amelie.fontaine@salesforward.eu",
     "su": "Feature Opportunity: Top 30 BD Professionals in Luxembourg for 2026",
     "bo": """<p>Dear Sales / Business Development Team,</p>
<p>My name is Am\u00e9lie Fontaine and I am a contributing editor at <strong>SalesForward</strong>, a European publication focused on B2B sales strategy and business development best practices.</p>
<p>Each year we compile our "Top 30 BD Professionals" list spotlighting outstanding business development talent across the Benelux region. LuxConsultancy has been nominated by two independent industry contacts, and we would love to feature one or more members of your team in this year's edition.</p>
<p>What this involves:</p>
<ul>
<li>A short Q&A interview (20\u201330 minutes, conducted via video call)</li>
<li>A professional profile published in our June digital edition (readership: ~18,000 BD/sales professionals)</li>
<li>Inclusion in our online directory with a backlink to your company website</li>
</ul>
<p>There is no cost or obligation \u2014 this is an editorial feature, not a sponsored placement. We are collecting nominations through the end of April, so there is plenty of time.</p>
<p>If anyone on your team would be interested, simply reply to this email and I will send over the details.</p>
<p>Looking forward to hearing from you.</p>
<p>Best regards,<br>
Am\u00e9lie Fontaine<br>
Contributing Editor, SalesForward<br>
amelie.fontaine@salesforward.eu<br>
+352 27 86 4410</p>"""},
]


OPS_EMAILS = [
    # Cell 9: Phish/Known/High/Reward — Spoofed wattvoltbrdige.com
    {"sn": "WattVoltBridge Procurement", "se": "savings@wattvoltbrdige.com",
     "su": "URGENT: Equipment Cost Savings Rebate - Claim €4,500 Now",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#2e7d32;color:white;padding:15px;"><h2 style="margin:0;">💰 Procurement Rebate Available</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Operations Team,</p>
<p>Based on your equipment purchases in Q4 2024, you qualify for a <strong>procurement rebate of €4,500</strong>.</p>
<div style="background:#e8f5e9;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:28px;font-weight:bold;color:#2e7d32;">€4,500 Rebate</span></div>
<p><strong>Claim must be submitted within 24 hours</strong> as per our quarterly rebate window.</p>
<p style="text-align:center;"><a href="http://wattvoltbridge-procurement.com/rebate/claim/luxconsultancy" style="background:#2e7d32;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Claim Rebate</a></p>
<p style="font-size:12px;color:#666;">Company banking details required for direct deposit.</p>
</div></div>"""},
    # Cell 10: Phish/Unknown/Low/Threat
    {"sn": "Supply Chain Audit Service", "se": "verify@supplychainaudit.net",
     "su": "Routine Supply Chain Compliance Verification",
     "bo": """<p>Dear LuxConsultancy Operations Manager,</p>
<p>Your organization has been selected for a routine supply chain compliance verification under EU Regulation 2024/1760 (Corporate Sustainability Due Diligence).</p>
<p><strong>Areas under review:</strong></p>
<ul><li>Supplier code of conduct documentation</li><li>Environmental impact assessments for key vendors</li><li>Conflict minerals due diligence records</li></ul>
<p>Please complete the pre-verification questionnaire:</p>
<p><a href="http://supplychainaudit.net/questionnaire/luxconsultancy" style="color:#0078d4;">Complete Questionnaire</a></p>
<p>The deadline is flexible — we recommend completing within 3 weeks.</p>
<p>Regards,<br>Supply Chain Audit Service<br>European Compliance Division</p>"""},
    # Cell 11: Phish/Known/Low/Threat — Spoofed secure-nebula.com
    {"sn": "SecureNebula Asset Management", "se": "assets@secure-nebula.com",
     "su": "Asset Tracking System - Credential Update Needed",
     "bo": """<p>Dear LuxConsultancy Operations Team,</p>
<p>We're updating the authentication module for our cloud-based asset tracking platform. To maintain access to your inventory management dashboard, please re-verify your credentials.</p>
<p><strong>What's changing:</strong></p>
<ul><li>Migration to OAuth 2.0 authentication</li><li>New API key generation required</li><li>Updated access permissions for team members</li></ul>
<p>Update your credentials through the asset management portal:</p>
<p><a href="http://secure-nebula-assets.com/credential-update" style="color:#0078d4;text-decoration:underline;">Update Asset Portal Credentials</a></p>
<p>If not updated within 2 weeks, access will default to read-only mode.</p>
<p>SecureNebula Asset Management<br>assets@secure-nebula.com</p>"""},
    # Cell 12: Phish/Unknown/High/Reward
    {"sn": "Industrial Excellence Awards", "se": "nominations@industrialawards-eu.com",
     "su": "URGENT: Best Operations Team Award - Nomination Closing Today",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#ffd700,#ff8c00);color:#1a1a1a;padding:20px;text-align:center;"><h1 style="margin:0;">🏆 NOMINATION ALERT</h1></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Operations Leadership,</p>
<p>Your operations team has been nominated for the <strong>2025 European Industrial Excellence Award</strong> in the category of:</p>
<div style="background:#fff8e1;padding:15px;text-align:center;margin:15px 0;border:2px solid #ffd700;">
<h3 style="margin:0;color:#ff8c00;">Best Operations Team - Process Innovation</h3></div>
<p style="color:#d32f2f;"><strong>Nomination closes today at 6 PM CET.</strong></p>
<p style="text-align:center;"><a href="http://industrialawards-eu.com/confirm/luxconsultancy-ops" style="background:#ff8c00;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Confirm Nomination</a></p>
<p style="font-size:12px;color:#666;">Includes invitation to the awards gala in Vienna.</p>
</div></div>"""},
    # Cell 13: Legit/Known/Low/Reward — Real wattvoltbridge.com
    {"sn": "WattVoltBridge Tools", "se": "tools@wattvoltbridge.com",
     "su": "New Energy Efficiency Tools for Project Managers - Free Access",
     "bo": """<p>Hi LuxConsultancy operations team,</p>
<p>We've launched a new suite of <strong>energy efficiency planning tools</strong> that we're offering free to our consulting partners for the first 6 months.</p>
<p><strong>New tools include:</strong></p>
<ul><li>Carbon footprint calculator for facility assessments</li><li>Energy audit template library (40+ templates)</li><li>ROI modeling for renewable energy transitions</li></ul>
<p>Access the toolkit: <a href="https://tools.wattvoltbridge.com/partners/luxconsultancy/energy-suite" style="color:#0078d4;">WattVoltBridge Energy Toolkit</a></p>
<p>No credit card required. We'd love your feedback during the trial period.</p>
<p>Best,<br>WattVoltBridge Product Team<br>tools@wattvoltbridge.com</p>"""},
    # Cell 14: Legit/Unknown/High/Threat
    {"sn": "EU Safety Standards Agency", "se": "regulations@safetystandards-eu.org",
     "su": "URGENT: Updated Workplace Safety Regulations Now in Effect",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#e65100;color:white;padding:15px;"><h2 style="margin:0;">⚠️ EU Safety Standards Agency</h2></div>
<div style="padding:20px;">
<p>Dear Operations / Facilities Manager,</p>
<p>Directive 2025/0089 on workplace safety in professional services environments <strong>is now in effect</strong>. Key changes affect consulting firms with client-facing office spaces.</p>
<p><strong>Mandatory requirements:</strong></p>
<ul><li>Updated fire evacuation plans (maximum 90-day review cycle)</li><li>Ergonomic workstation assessments for all employees</li><li>Mental health first aider designation (1 per 50 employees)</li></ul>
<p><strong style="color:#d32f2f;">Compliance deadline: 30 days from effective date. Penalties: up to €10,000 per violation.</strong></p>
<p>Access compliance resources: <a href="https://safetystandards-eu.org/directives/2025-0089/employer-toolkit" style="color:#0078d4;text-decoration:none;">Employer Compliance Toolkit</a></p>
<p>EU Safety Standards Agency<br>Avenue de Cortenbergh 100, Brussels</p>
</div></div>"""},
    # Cell 15: Legit/Known/High/Threat — Real greenenvi.com
    {"sn": "Sophie Laurent (GreenEnvi)", "se": "s.laurent@greenenvi.com",
     "su": "URGENT: Environmental Compliance Deadline for Current Project",
     "bo": """<p>Hi team,</p>
<p>Quick urgent update — the environmental impact assessment for the WattVoltBridge renewable energy project <strong>must be submitted to the regulatory authority by this Friday</strong>.</p>
<p><strong>Outstanding items from LuxConsultancy's side:</strong></p>
<ol><li>Finalized carbon offset calculations for Phase 2</li><li>Signed stakeholder consultation summary</li><li>Updated biodiversity impact matrix</li></ol>
<p><strong style="color:#d32f2f;">If these aren't submitted by Friday 5 PM CET, the project permit application will be delayed by at least 8 weeks.</strong></p>
<p>I've shared the submission checklist here: <a href="https://docs.greenenvi.com/shared/wvb-eia-checklist-2025" style="color:#0078d4;text-decoration:none;">EIA Submission Checklist</a></p>
<p>Please prioritize and let me know if you need anything from our side.</p>
<p>Thanks,<br>Sophie Laurent<br>Lead Environmental Consultant, GreenEnvi<br>+31 621 319 888</p>"""},
    # Cell 16: Legit/Unknown/Low/Reward
    {"sn": "Supply Chain Council EU", "se": "events@supplychaincouncil-eu.org",
     "su": "Free Supply Chain Innovation Conference - Registration Open",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#00695c;color:white;padding:15px;"><h2 style="margin:0;">Supply Chain Council EU</h2></div>
<div style="padding:20px;">
<p>Dear Operations Professional,</p>
<p>Join us at the <strong>European Supply Chain Innovation Conference 2025</strong>, free for qualifying organizations.</p>
<div style="background:#e0f2f1;padding:15px;margin:15px 0;border-left:4px solid #00695c;">
<p style="margin:0;"><strong>Theme:</strong> Digital Twins and AI in Supply Chain Management</p>
<p style="margin:5px 0;"><strong>Date:</strong> April 3-4, 2025</p>
<p style="margin:0;"><strong>Venue:</strong> RAI Convention Centre, Amsterdam</p></div>
<p>Featuring keynotes from DHL, Siemens, and Maersk on next-generation logistics.</p>
<p style="text-align:center;"><a href="https://supplychaincouncil-eu.org/conference-2025/register" style="background:#00695c;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Free</a></p>
</div></div>"""},
]

CUSTOMER_SERVICE_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "LuxConsultancy HR Rewards", "se": "rewards@luxconsultancy.net",
     "su": "You've Been Nominated for the Q1 Customer Excellence Award \u2014 Action Required by EOD",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#1a3c6e;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">LuxConsultancy Customer Excellence Program</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;">
<p>Dear Team Member,</p>
<p>Congratulations! Based on your outstanding client satisfaction scores this quarter, you have been nominated for the <strong>Q1 Customer Excellence Award</strong>. This recognition comes with a <strong>\u20ac500 bonus</strong> and a featured profile on the company intranet.</p>
<p>To confirm your nomination and claim your reward, you must complete the acceptance form before <strong>end of business today</strong>. Unclaimed nominations will be reassigned to the next eligible candidate.</p>
<p style="text-align:center;margin:25px 0;">
<a href="http://luxconsultancy-rewards.net/excellence-award/confirm" style="background-color:#1a3c6e;color:#ffffff;padding:12px 30px;text-decoration:none;border-radius:4px;font-weight:bold;">Accept Your Nomination</a>
</p>
<p>Please have your employee ID and CRM login credentials ready for verification purposes.</p>
<p>Best regards,<br>Employee Recognition Team<br>LuxConsultancy Human Resources</p>
</div></div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "ClientPulse Analytics", "se": "alerts@clientpulse-analytics.com",
     "su": "Negative Client Sentiment Detected \u2014 Review Required",
     "bo": """<p>Hello,</p>
<p>Our sentiment analysis platform has flagged a pattern of declining client satisfaction scores associated with your support queue at LuxConsultancy. Over the past 30 days, the following trends were identified:</p>
<ul>
<li>Average resolution satisfaction down 12% compared to previous quarter</li>
<li>3 escalation cases flagged for potential SLA non-compliance</li>
<li>Client retention risk score elevated for 2 key accounts</li>
</ul>
<p>These findings will be included in the next quarterly performance review cycle unless addressed. You can view the full report and submit contextual notes through our portal:</p>
<p><a href="http://clientpulse-analytics.com/report/luxconsultancy/cs-review" style="color:#0078d4;text-decoration:underline;">View Sentiment Analysis Report</a></p>
<p>You will need to authenticate with your company email credentials to access account-specific data.</p>
<p>Regards,<br>ClientPulse Analytics Team</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "SecureNebula Cloud Services", "se": "support@secure-nebula.com",
     "su": "CRM Data Retention Policy Update \u2014 Compliance Action Needed",
     "bo": """<p>Dear LuxConsultancy Support Team,</p>
<p>As part of our updated data retention policy effective April 15, 2026, all client interaction records stored on SecureNebula's cloud infrastructure must be re-certified by account administrators.</p>
<p>Accounts that are not re-certified may experience the following:</p>
<ul>
<li>Temporary suspension of CRM data sync services</li>
<li>Archived client records moved to cold storage (retrieval delays up to 48 hours)</li>
<li>Potential gaps in support ticket history</li>
</ul>
<p>This is a routine compliance step and should only take a few minutes. Please log in to the admin portal to complete the certification:</p>
<p><a href="http://secure-nebula.com/admin/data-retention/certify" style="color:#0078d4;text-decoration:underline;">Complete Data Retention Certification</a></p>
<p>If you have questions, contact our support team at compliance@secure-nebula.com.</p>
<p>Thank you,<br>SecureNebula Compliance Team</p>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "European Customer Service Institute", "se": "awards@ecsi-europe.org",
     "su": "URGENT: LuxConsultancy Shortlisted for 2026 Service Excellence Award \u2014 Response Deadline Today",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#2e7d32;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">European Customer Service Institute</h2>
<p style="color:#c8e6c9;margin:5px 0 0 0;font-size:14px;">2026 Service Excellence Awards</p>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;">
<p>Dear LuxConsultancy Customer Service Team,</p>
<p>We are pleased to inform you that <strong>LuxConsultancy</strong> has been shortlisted for the <strong>2026 European Service Excellence Award</strong> in the Consulting &amp; Professional Services category.</p>
<p>As a shortlisted organisation, you are entitled to:</p>
<ul>
<li>Complimentary attendance for two team members at the Awards Gala in Brussels (valued at \u20ac1,200 per seat)</li>
<li>Featured profile in our annual European Service Leaders publication</li>
<li>Use of the ECSI Shortlisted badge on your company website and materials</li>
</ul>
<p>Due to the volume of nominations this year, we require confirmation from a designated team representative by <strong>5:00 PM CET today</strong>. Unconfirmed shortlistings will be forfeited and offered to the next eligible firm.</p>
<p style="text-align:center;margin:25px 0;">
<a href="http://ecsi-europe.org/awards/2026/confirm-shortlist" style="background-color:#2e7d32;color:#ffffff;padding:12px 30px;text-decoration:none;border-radius:4px;font-weight:bold;">Confirm Your Shortlisting</a>
</p>
<p>Please have your business email and company registration details available for the verification step.</p>
<p>With kind regards,<br>Dr. Annelise Richter<br>Awards Programme Director<br>European Customer Service Institute</p>
</div></div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "TrendyLetter", "se": "digest@trendyletter.com",
     "su": "This Month in CX: Top Strategies That Boosted Client Retention by 20%",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#6a1b9a;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">TrendyLetter \u2014 CX Professional Edition</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;">
<p>Hi there,</p>
<p>Your March edition of the Customer Experience Professional digest is ready. Here are this month's highlights:</p>
<ul>
<li><strong>Case Study:</strong> How a mid-size consulting firm reduced churn by 20% with proactive outreach</li>
<li><strong>Tool Review:</strong> 5 CRM integrations that streamline ticket escalation workflows</li>
<li><strong>Industry Insight:</strong> The growing role of AI-assisted sentiment analysis in client support</li>
</ul>
<p>Plus, subscribers who complete our short readership survey will be entered into a draw for a <strong>\u20ac100 gift voucher</strong>.</p>
<p><a href="https://trendyletter.com/cx-digest/march-2026" style="color:#0078d4;text-decoration:underline;">Read the Full Digest</a></p>
<p>Happy reading,<br>The TrendyLetter Team</p>
</div></div>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "GDPR Compliance Bureau \u2014 Luxembourg", "se": "notifications@cnpd-luxembourg.lu",
     "su": "URGENT: Client Data Subject Access Request \u2014 72-Hour Response Window",
     "bo": """<p>Dear Data Processing Contact at LuxConsultancy,</p>
<p>We are writing to formally notify you that a Data Subject Access Request (DSAR) has been submitted to our office referencing personal data held by LuxConsultancy in connection with consulting services provided between June and December 2025.</p>
<p>Under GDPR Article 15, your organisation is required to respond within <strong>30 calendar days</strong> of the original request date. As this request was filed on 1 March 2026, the statutory deadline is <strong>31 March 2026</strong>. Given the approaching deadline, we urge prompt action.</p>
<p>The following steps are required:</p>
<ul>
<li>Acknowledge receipt of this notification within 72 hours via the secure portal</li>
<li>Identify and compile all personal data records pertaining to the data subject</li>
<li>Submit your formal response through the CNPD case management system</li>
</ul>
<p>You can access the full request details and respond here:<br>
<a href="https://cnpd-luxembourg.lu/dsar/case/2026-LUX-04417" style="color:#0078d4;text-decoration:underline;">CNPD Case Portal \u2014 Reference 2026-LUX-04417</a></p>
<p>Failure to comply within the statutory timeframe may result in administrative proceedings.</p>
<p>Kind regards,<br>Sophie Laurent<br>Data Protection Officer \u2014 Case Management<br>Commission Nationale pour la Protection des Donn\u00e9es<br>Luxembourg</p>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "LockGrid Security Solutions", "se": "alerts@lockgrid.com",
     "su": "CRITICAL: Unusual Login Activity on Your CRM Support Console",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#c62828;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">\u26a0 Security Alert \u2014 LockGrid</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;">
<p>Dear LuxConsultancy Security Contact,</p>
<p>Our monitoring systems have detected unusual login activity on the CRM support console protected by LockGrid endpoint security. The details are as follows:</p>
<div style="background-color:#fff3e0;border-left:4px solid #e65100;padding:12px;margin:15px 0;">
<strong>Alert Details:</strong><br>
Timestamp: 28 March 2026, 03:42 AM CET<br>
Source IP: 185.220.101.xx (TOR exit node)<br>
Target: CRM Admin Console \u2014 client records module<br>
Status: Blocked by LockGrid policy
</div>
<p>While the attempt was <strong>blocked</strong>, we strongly recommend the following immediate actions:</p>
<ul>
<li>Review active sessions and revoke any unrecognised access tokens</li>
<li>Enable enhanced MFA on all admin-level CRM accounts</li>
<li>Contact your LockGrid account manager to schedule a security audit</li>
</ul>
<p>You can review the full incident log in your LockGrid dashboard:<br>
<a href="https://lockgrid.com/dashboard/incidents/2026-03-28-LUX" style="color:#0078d4;text-decoration:underline;">View Incident Report</a></p>
<p>If you did not recognise this activity, please respond to this email or call our 24/7 security hotline at +352 27 44 88 00.</p>
<p>Regards,<br>LockGrid Threat Response Team</p>
</div></div>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "ServiceBench Research", "se": "invitations@servicebench-research.com",
     "su": "Invitation: Participate in Our 2026 Client Support Benchmarking Study (\u20ac75 Incentive)",
     "bo": """<p>Dear Customer Service Professional,</p>
<p>ServiceBench Research is conducting our annual <strong>European Client Support Benchmarking Study</strong>, and we would like to invite LuxConsultancy to participate. This study gathers anonymised operational data from consulting firms across Europe to produce industry-wide service quality benchmarks.</p>
<p>Participation involves:</p>
<ul>
<li>A 15-minute online questionnaire covering support volume, resolution times, and client satisfaction metrics</li>
<li>Optional follow-up interview (30 minutes) for deeper insights</li>
</ul>
<p>In return, participating organisations receive:</p>
<ul>
<li>A complimentary copy of the 2026 Benchmarking Report (normally \u20ac450)</li>
<li>A \u20ac75 Amazon voucher for each completed questionnaire</li>
<li>Early access to anonymised peer comparison data</li>
</ul>
<p>There is no deadline pressure \u2014 the study remains open through June 2026. You can learn more and register your interest here:</p>
<p><a href="https://servicebench-research.com/benchmarking/2026/register" style="color:#0078d4;text-decoration:underline;">Learn More &amp; Register</a></p>
<p>Best regards,<br>Dr. Marcus Engel<br>Lead Researcher \u2014 Service Operations<br>ServiceBench Research, Frankfurt</p>"""},
]

MARKETING_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "TrendyLetter Campaign Awards", "se": "awards@trendyletter.net",
     "su": "URGENT: You've Won the 2025 Best Campaign Award \u2014 Claim Your \u20ac2,500 Prize",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#7c3aed,#a855f7);color:white;padding:20px;text-align:center;"><h1 style="margin:0;">\U0001F3C6 2025 Campaign Excellence Award</h1>
<p style="margin:5px 0 0 0;font-size:14px;">TrendyLetter Annual Recognition Program</p></div>
<div style="padding:25px;">
<p>Dear LuxConsultancy Marketing Team,</p>
<p>Congratulations! Your Q4 newsletter campaign <em>"Sustainable Consulting Trends"</em> has been selected as the <strong>Best Performing Campaign of 2025</strong> across all TrendyLetter enterprise accounts.</p>
<div style="background:#faf5ff;padding:20px;margin:20px 0;border-left:4px solid #7c3aed;text-align:center;">
<p style="margin:0;font-size:14px;color:#6b21a8;">Your prize</p>
<span style="font-size:32px;font-weight:bold;color:#7c3aed;">\u20ac2,500 Marketing Credit</span>
<p style="margin:8px 0 0 0;font-size:13px;color:#666;">Redeemable for premium campaign placements &amp; analytics tools</p></div>
<p style="color:#d32f2f;"><strong>You must claim your award within 12 hours or the prize will be allocated to the runner-up.</strong></p>
<p style="text-align:center;"><a href="http://trendyletter-awards.net/claim/luxconsultancy-2025" style="background:#7c3aed;color:white !important;padding:14px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Claim Your Award Now</a></p>
<p style="font-size:12px;color:#666;">You will need to verify your account credentials and provide campaign manager details to process the credit.</p>
</div></div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "BrandGuard Compliance", "se": "alerts@brandguard-monitor.com",
     "su": "Trademark Infringement Notice - LuxConsultancy Brand Assets",
     "bo": """<p>Dear LuxConsultancy Marketing Department,</p>
<p>During a routine scan of digital advertising networks, our brand protection team identified potential unauthorized use of LuxConsultancy trademarked materials on third-party websites.</p>
<p><strong>Summary of findings:</strong></p>
<ul><li>3 instances of logo misuse on affiliate advertising sites</li><li>1 case of slogan reproduction without licensing</li><li>Unauthorized use of brand color palette in competitor materials</li></ul>
<p>We recommend reviewing the detailed report and submitting a formal response at your convenience to protect your intellectual property rights.</p>
<p><a href="http://brandguard-monitor.com/reports/luxconsultancy-tm-2025" style="color:#0078d4;text-decoration:underline;">View Full Infringement Report</a></p>
<p>If left unaddressed, these infringements could dilute brand recognition over time. No immediate deadline applies, but we suggest reviewing within the next few weeks.</p>
<p>Best regards,<br>Ana Petrova<br>Brand Protection Analyst<br>BrandGuard Compliance Services</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "SecureNebula Cloud Services", "se": "cloud-ops@securenebula.org",
     "su": "Marketing Cloud Storage Approaching Quota Limit",
     "bo": """<p>Hello LuxConsultancy Marketing Team,</p>
<p>This is a routine notification that your Marketing department cloud storage is at 87% capacity. If storage reaches 100%, new file uploads (including campaign assets and brand materials) will be blocked.</p>
<p><strong>Current usage breakdown:</strong></p>
<ul><li>Campaign image library: 42 GB</li><li>Video assets: 31 GB</li><li>Brand guidelines &amp; templates: 8 GB</li><li>Archived campaigns: 19 GB</li></ul>
<p>You can free up space or request a quota increase through our portal:</p>
<p><a href="http://securenebula-cloud.org/storage/manage?client=luxconsultancy" style="color:#0078d4;">Manage Storage Quota</a></p>
<p>There is no immediate rush, but we recommend addressing this before your next major campaign launch to avoid disruption.</p>
<p>Kind regards,<br>SecureNebula Cloud Operations<br>support@securenebula.org</p>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "DigiMarketer Pro", "se": "partnerships@digimarketer-pro.com",
     "su": "URGENT: Exclusive Sponsorship Slot for LuxConsultancy \u2014 Response Needed Today",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#0891b2;color:white;padding:15px;"><h2 style="margin:0;">DigiMarketer Pro \u2014 Partner Opportunity</h2></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Marketing Director,</p>
<p>We have an exclusive, time-sensitive opportunity for your brand. A premium sponsor has withdrawn from our <strong>European Digital Marketing Summit 2025</strong>, and we are offering the slot to select consulting firms.</p>
<div style="background:#ecfeff;padding:20px;margin:20px 0;border:2px solid #0891b2;border-radius:8px;text-align:center;">
<p style="margin:0;font-size:14px;">Platinum Sponsorship Package</p>
<p style="margin:8px 0;font-size:28px;font-weight:bold;color:#0891b2;">75% Off \u2014 \u20ac4,750 (was \u20ac19,000)</p>
<p style="margin:0;font-size:13px;color:#666;">Includes keynote slot, branded lounge, 10,000+ attendee exposure</p></div>
<p style="color:#d32f2f;"><strong>This offer expires at 5:00 PM CET today. Another firm is already reviewing the package.</strong></p>
<p style="text-align:center;"><a href="http://digimarketer-pro.com/sponsors/platinum-offer/luxconsultancy" style="background:#0891b2;color:white !important;padding:14px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Reserve Sponsorship Slot</a></p>
<p style="font-size:12px;color:#666;">Company billing details and authorized signatory information required to confirm.</p>
</div></div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "Sophie Laurent (GreenEnvi)", "se": "s.laurent@greenenvi.com",
     "su": "Free Co-Branding Opportunity \u2014 Joint Sustainability Report",
     "bo": """<p>Hi Marketing Team,</p>
<p>Hope you're all doing well! I wanted to share an exciting opportunity with you.</p>
<p>GreenEnvi is publishing our annual <strong>European Sustainability in Business Report</strong> this spring, and we'd love to feature LuxConsultancy as a co-branded partner. It's a great way to showcase your firm's commitment to sustainable consulting practices.</p>
<p><strong>What's included for partners:</strong></p>
<ul><li>Your logo and a 500-word company spotlight in the report</li><li>Social media cross-promotion (our audience: 45K+ sustainability professionals)</li><li>Complimentary access to all report data and infographics for your own campaigns</li></ul>
<p>No cost involved \u2014 we just need your brand assets and a short write-up. Take a look at last year's edition for reference: <a href="https://reports.greenenvi.com/sustainability-business-2024" style="color:#0078d4;text-decoration:none;">2024 Sustainability Report</a></p>
<p>No rush at all \u2014 just let me know if you're interested and I'll send over the details.</p>
<p>Best,<br>Sophie Laurent<br>Sustainability Consultant<br>GreenEnvi Solutions<br>+31 621 319 888</p>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "EU Advertising Standards Authority", "se": "compliance@eu-asa.europa.eu",
     "su": "URGENT: Consumer Complaint Filed Against LuxConsultancy Ad Campaign \u2014 Response Due in 5 Days",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1e3a5f;color:white;padding:15px;"><h2 style="margin:0;">EU Advertising Standards Authority</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Consumer Protection &amp; Advertising Compliance</p></div>
<div style="padding:20px;">
<p><strong>Reference:</strong> EU-ASA/2025/COMP-0847</p>
<p>Dear LuxConsultancy Marketing Compliance Officer,</p>
<p>We have received a formal consumer complaint regarding a digital advertising campaign attributed to LuxConsultancy, running across programmatic display networks in France and Germany.</p>
<p><strong>Complaint details:</strong></p>
<ul><li>Campaign: "Transform Your Business with LuxConsultancy" (Display &amp; Social)</li><li>Allegation: Misleading performance claims under EU Directive 2005/29/EC</li><li>Complainant: Filed anonymously through our consumer portal</li></ul>
<p><strong style="color:#d32f2f;">A formal response must be submitted within 5 business days to avoid escalation to enforcement proceedings.</strong></p>
<p>Please review the full complaint and submit your response via our secure portal:</p>
<p><a href="https://complaints.eu-asa.europa.eu/cases/2025-COMP-0847/respond" style="color:#1e3a5f;font-weight:bold;text-decoration:underline;">Access Complaint &amp; Response Portal</a></p>
<p>For procedural questions, contact our compliance helpline: +32 2 299 11 11</p>
<p>Regards,<br>EU Advertising Standards Authority<br>Rue de la Loi 200, 1049 Brussels</p>
</div></div>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "Marcus Chen (LockGrid)", "se": "m.chen@lockgrid.com",
     "su": "URGENT: Compromised Social Media Credentials \u2014 Immediate Password Reset Required",
     "bo": """<p>Hi Marketing Team,</p>
<p>We need your attention on this right away. During our routine monitoring of LuxConsultancy's digital footprint, we detected that credentials for your corporate social media management platform appear in a recently published data breach dump.</p>
<p><strong>What we found:</strong></p>
<ol><li>Email/password combinations matching your Hootsuite enterprise account</li><li>An API token associated with your LinkedIn company page</li><li>Session cookies from your Buffer scheduling tool</li></ol>
<p><strong style="color:#d32f2f;">Please reset all social media account passwords and revoke active API tokens by end of day today to prevent unauthorized posting or data access.</strong></p>
<p>I've compiled the full details and recommended remediation steps in our incident portal:</p>
<p><a href="https://portal.lockgrid.com/clients/luxconsultancy/incident-2025-0341" style="color:#0078d4;text-decoration:none;">View Incident Report &amp; Remediation Steps</a></p>
<p>Let me know once the resets are done so we can verify the accounts are secure.</p>
<p>Best regards,<br>Marcus Chen<br>Senior Security Consultant<br>LockGrid Solutions<br>Direct: +31 26 48 71 33</p>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "European Content Marketing Association", "se": "membership@ecma-europe.org",
     "su": "Complimentary ECMA Membership for LuxConsultancy Marketing Team",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#059669;color:white;padding:15px;"><h2 style="margin:0;">European Content Marketing Association</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Empowering Marketing Professionals Across Europe</p></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Marketing Team,</p>
<p>The European Content Marketing Association (ECMA) is pleased to offer your team a <strong>complimentary one-year professional membership</strong> as part of our initiative to support marketing excellence in the consulting sector.</p>
<div style="background:#ecfdf5;padding:15px;margin:15px 0;border-left:4px solid #059669;">
<p style="margin:0;"><strong>Membership benefits include:</strong></p>
<ul style="margin:10px 0 0 0;padding-left:20px;">
<li>Access to our library of 200+ marketing case studies and benchmarks</li>
<li>Monthly webinars with industry thought leaders</li>
<li>Discounted rates for ECMA annual conference in Barcelona</li>
<li>Networking directory of 5,000+ European marketing professionals</li></ul></div>
<p>There is no obligation and no payment required. Simply register your team when convenient:</p>
<p style="text-align:center;"><a href="https://ecma-europe.org/membership/register?ref=consulting-outreach" style="background:#059669;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register for Free Membership</a></p>
<p style="font-size:12px;color:#666;">This offer is open-ended \u2014 register whenever suits your schedule.</p>
</div></div>"""},
]

PROCUREMENT_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "Procurement Awards - LuxConsultancy", "se": "procurement-awards@luxconsultancy.net",
     "su": "URGENT: You've Won the Q1 Preferred Vendor Negotiation Bonus - Claim by EOD",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#1565c0,#0d47a1);color:white;padding:20px;text-align:center;">
<h2 style="margin:0;">Procurement Excellence Award</h2>
<p style="margin:5px 0 0 0;font-size:13px;">LuxConsultancy Internal Recognition Program</p></div>
<div style="padding:20px;">
<p>Dear Procurement Team Member,</p>
<p>Congratulations! Based on your outstanding vendor negotiation results in Q1 2025, you have been selected to receive a <strong>\u20ac750 performance bonus</strong> under the Procurement Excellence Initiative.</p>
<div style="background:#e3f2fd;padding:15px;text-align:center;margin:15px 0;border-radius:8px;border:2px solid #1565c0;">
<span style="font-size:28px;font-weight:bold;color:#1565c0;">\u20ac750.00 Bonus</span><br>
<span style="font-size:13px;color:#555;">Deposited directly to your payroll account</span></div>
<p style="color:#d32f2f;"><strong>You must confirm your banking details by end of business today</strong> to be included in this month's payroll cycle. Unclaimed bonuses will be forfeited.</p>
<p style="text-align:center;"><a href="http://luxconsultancy-awards.net/claim-bonus/procurement-q1" style="background:#1565c0;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Confirm &amp; Claim Bonus</a></p>
<p style="font-size:12px;color:#666;">Ref: PROC-AWARD-2025-Q1-0847</p>
</div></div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "VendorShield Compliance", "se": "alerts@vendorshield-compliance.net",
     "su": "Vendor Compliance Certification Gap Identified",
     "bo": """<p>Dear Procurement Manager,</p>
<p>During a routine audit of supplier compliance records across European consulting firms, our system flagged LuxConsultancy for a potential gap in vendor certification documentation.</p>
<p><strong>Issue identified:</strong></p>
<ul><li>3 active vendors missing updated ISO 27001 certificates</li><li>2 suppliers with expired liability insurance documentation</li><li>1 subcontractor without current GDPR processing agreement</li></ul>
<p>While this is not yet critical, unresolved gaps may affect your eligibility for upcoming public sector tenders.</p>
<p>Review your vendor compliance status and upload missing documents here:</p>
<p><a href="http://vendorshield-compliance.net/portal/luxconsultancy/review" style="color:#0078d4;">Access Compliance Review Portal</a></p>
<p>Please address these findings at your earliest convenience.</p>
<p>Regards,<br>VendorShield Compliance Services<br>European Procurement Standards Division</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "SecureNebula Procurement Portal", "se": "procurement-support@securenebula.org",
     "su": "Supplier Portal Credential Rotation - Update Required",
     "bo": """<p>Dear LuxConsultancy Purchasing Team,</p>
<p>As part of our annual security policy update, all supplier portal credentials are being rotated. Your current login for the SecureNebula Vendor Management System will expire at the end of this month.</p>
<p><strong>What you need to do:</strong></p>
<ul><li>Re-verify your procurement team's portal access credentials</li><li>Update your organization's billing contact information</li><li>Confirm your approved purchaser list</li></ul>
<p>If credentials are not updated before expiry, your team will temporarily lose access to purchase order tracking and invoice submission features.</p>
<p>Update your credentials here: <a href="http://securenebula-portal.org/vendor-credential-reset" style="color:#1a73e8;text-decoration:underline;">Credential Reset Portal</a></p>
<p>This is a routine procedure. No disruption to existing orders is expected if completed on time.</p>
<p>Best regards,<br>SecureNebula Procurement Support<br>vendor-support@securenebula.org</p>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "GlobalSource Procurement Network", "se": "partnerships@globalsource-procurement.com",
     "su": "URGENT: Exclusive Bulk Discount Agreement - 35% Savings Expiring Tomorrow",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#2e7d32,#1b5e20);color:white;padding:20px;text-align:center;">
<h1 style="margin:0;font-size:22px;">Exclusive Partner Offer</h1>
<p style="margin:5px 0 0 0;">GlobalSource Procurement Network</p></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Procurement Director,</p>
<p>Based on your firm's purchasing volume, LuxConsultancy has been pre-approved for our <strong>Elite Partner Bulk Discount Program</strong>, offering up to <strong>35% savings</strong> on IT hardware, office supplies, and professional services.</p>
<div style="background:#e8f5e9;padding:20px;text-align:center;margin:15px 0;border:2px solid #4caf50;border-radius:8px;">
<span style="font-size:32px;font-weight:bold;color:#2e7d32;">Save up to 35%</span><br>
<span style="color:#555;">Estimated annual savings: \u20ac42,000+</span></div>
<p style="color:#d32f2f;"><strong>This pre-approved rate expires tomorrow at 5:00 PM CET.</strong> After this window, standard pricing applies and re-qualification takes 90 days.</p>
<p style="text-align:center;"><a href="http://globalsource-procurement.com/elite-partner/activate/luxconsultancy" style="background:#2e7d32;color:white !important;padding:14px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Activate Partner Discount</a></p>
<p style="font-size:12px;color:#666;">You will need to verify your company details and authorized signatory to activate the agreement.</p>
</div></div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "Anne Richter (SuperFinance)", "se": "a.richter@superfinance.com",
     "su": "New Procurement Cost Analytics Dashboard Available",
     "bo": """<p>Hi team,</p>
<p>Just a heads-up that we've launched a new <strong>Procurement Cost Analytics Dashboard</strong> as part of our partnership with LuxConsultancy. It's now available in your SuperFinance portal.</p>
<p><strong>What's included:</strong></p>
<ul><li>Real-time spend tracking across all vendor categories</li><li>Automated savings opportunity identification</li><li>Quarterly benchmark reports vs. industry averages</li><li>Export-ready data for budget presentations</li></ul>
<p>You can explore it at your convenience here: <a href="https://portal.superfinance.com/analytics/procurement-dashboard" style="color:#0078d4;text-decoration:none;">Open Procurement Dashboard</a></p>
<p>No rush on this \u2014 it's already populated with your latest data. Happy to walk you through it on our next scheduled call if you'd like.</p>
<p>Best regards,<br>Anne Richter<br>Client Analytics, SuperFinance<br>+352 27 86 41 15</p>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "EU Public Procurement Regulatory Office", "se": "compliance@euprocurement-authority.eu",
     "su": "URGENT: Updated Procurement Thresholds - Mandatory Compliance by April 15",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1a237e;color:white;padding:15px;"><h2 style="margin:0;">EU Public Procurement Regulatory Office</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Directorate for Procurement Standards</p></div>
<div style="padding:20px;">
<p><strong>Reference:</strong> EUPRO/2025/DIR-0391</p>
<p>Dear Procurement Officer,</p>
<p>Effective April 15, 2025, revised EU procurement directive thresholds will come into force. All entities engaged in public and semi-public procurement must update their internal policies accordingly.</p>
<p><strong>Key changes:</strong></p>
<ul><li>New threshold for supplies and services contracts: \u20ac143,000 (previously \u20ac140,000)</li><li>Revised rules for framework agreements exceeding 4-year terms</li><li>Mandatory e-procurement for all contracts above \u20ac25,000</li></ul>
<p><strong style="color:#d32f2f;">Non-compliance after April 15 may result in contract challenges and exclusion from public tenders.</strong></p>
<p>Download the full directive and implementation guidance: <a href="https://euprocurement-authority.eu/directives/2025-thresholds-update" style="color:#0078d4;text-decoration:none;">2025 Threshold Update (PDF)</a></p>
<p>For questions, contact our helpdesk: +32 2 299 1111</p>
<p>Regards,<br>EU Public Procurement Regulatory Office<br>Rue de la Loi 170, B-1048 Brussels</p>
</div></div>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "Thomas Meier (GreenEnvi)", "se": "t.meier@greenenvi.com",
     "su": "URGENT: Supply Chain Due Diligence Act - Vendor Audit Deadline Friday",
     "bo": """<p>Hi Procurement Team,</p>
<p>Quick urgent note \u2014 as you may know, the <strong>EU Corporate Sustainability Due Diligence Directive (CS3D)</strong> reporting deadline is this Friday, and we still need completed environmental compliance questionnaires from 4 of your Tier 1 suppliers.</p>
<p><strong>Missing responses from:</strong></p>
<ul><li>DataFlow Systems (IT infrastructure)</li><li>Meridian Office Solutions (office supplies)</li><li>ClearPath Logistics (courier services)</li><li>Alpine Facilities Management (cleaning &amp; maintenance)</li></ul>
<p><strong style="color:#d32f2f;">If these are not submitted by Friday 5 PM, LuxConsultancy's supply chain compliance report will be flagged as incomplete, which could affect your ESG rating.</strong></p>
<p>I've prepared the follow-up templates \u2014 grab them from our shared workspace: <a href="https://docs.greenenvi.com/shared/cs3d-vendor-templates-2025" style="color:#0078d4;text-decoration:none;">CS3D Vendor Questionnaire Templates</a></p>
<p>Let me know if you need help chasing any of these vendors. Happy to jump on a call.</p>
<p>Best,<br>Thomas Meier<br>Supply Chain Sustainability Lead<br>GreenEnvi Solutions<br>+49 170 554 2238</p>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "ProcureTech Innovation Forum", "se": "invitations@procuretech-forum.eu",
     "su": "Invitation: Free Webinar - AI-Powered Procurement Optimization",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#37474f;color:white;padding:15px;"><h2 style="margin:0;">ProcureTech Innovation Forum</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Advancing Procurement Through Technology</p></div>
<div style="padding:20px;">
<p>Dear Procurement Professional,</p>
<p>We'd like to invite LuxConsultancy to our upcoming complimentary webinar on <strong>AI-Powered Procurement Optimization</strong>.</p>
<div style="background:#f5f5f5;padding:15px;margin:15px 0;border-left:4px solid #37474f;">
<p style="margin:0;"><strong>Date:</strong> April 24, 2025 | 2:00 PM CET</p>
<p style="margin:5px 0;"><strong>Duration:</strong> 45 minutes + Q&amp;A</p>
<p style="margin:0;"><strong>Cost:</strong> <span style="color:#2e7d32;font-weight:bold;">Complimentary</span></p></div>
<p><strong>Topics covered:</strong></p>
<ul><li>Using machine learning for spend analysis and vendor scoring</li><li>Automating RFQ generation and bid comparison</li><li>Case study: How a European consultancy reduced procurement costs by 22%</li></ul>
<p>Register at your convenience: <a href="https://procuretech-forum.eu/webinars/ai-procurement-2025/register" style="color:#0078d4;text-decoration:none;">Reserve Your Spot</a></p>
</div></div>"""},
]

ADMIN_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "Claire Dubois", "se": "claire.dubois@luxconsultancy.net",
     "su": "URGENT: You've Been Selected for the Executive Assistant Recognition Award",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#1a3c6e;padding:20px;text-align:center;">
<h2 style="color:#ffffff;margin:0;">LuxConsultancy HR Recognition Programme</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;">
<p>Dear Colleague,</p>
<p>I am pleased to inform you that you have been nominated by your supervising partner for the <strong>Q1 2026 Executive Support Excellence Award</strong>. This recognition comes with a <strong>\u20ac750 bonus</strong> and a certificate presented at next week's town hall.</p>
<p>To accept your nomination and confirm your attendance at the ceremony, you must complete the acknowledgement form <strong>by end of business today</strong>. Delayed responses will forfeit the nomination to the next candidate on the list.</p>
<p style="text-align:center;margin:25px 0;">
<a href="http://luxconsultancy-recognition.net/award-accept?ref=ea2026q1" style="background-color:#1a3c6e;color:#ffffff;padding:12px 30px;text-decoration:none;border-radius:4px;font-weight:bold;">Accept Nomination &amp; Confirm</a>
</p>
<p>Please ensure your direct deposit details are current so the bonus can be processed with your March payroll.</p>
<p>Warm regards,<br>Claire Dubois<br>Head of People &amp; Culture<br>LuxConsultancy</p>
</div></div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "Facilities Alert - Regus Luxembourg", "se": "no-reply@regus-notifications.com",
     "su": "Action Needed: Building Access Card Revalidation",
     "bo": """<p>Hello,</p>
<p>As part of ongoing security compliance at the Regus Centre Luxembourg-Kirchberg, all tenant access badges must be revalidated before the end of this quarter.</p>
<p>Failure to revalidate your badge may result in temporary deactivation of your access privileges. This is a routine procedure and should only take a few minutes.</p>
<p>Please revalidate your badge by uploading a photo of your current card and confirming your personal details:</p>
<p><a href="http://regus-badge-portal.com/revalidate?tenant=luxconsultancy">http://regus-badge-portal.com/revalidate?tenant=luxconsultancy</a></p>
<p>If you have questions, contact our facilities desk at support@regus-notifications.com.</p>
<p>Thank you,<br>Regus Facilities Management<br>Luxembourg-Kirchberg Centre</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "SecureNebula Support", "se": "admin-alerts@secure-nebula.com",
     "su": "Document Storage Quota Warning - Action Required",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<p>Dear LuxConsultancy Administrator,</p>
<p>Our records show that your organisation's shared document storage on SecureNebula Cloud is approaching <strong>92% capacity</strong>. If storage reaches 100%, new file uploads will be blocked and auto-sync for executive calendars will be suspended.</p>
<p>We recommend reviewing your stored files and archiving older documents to free up space. Alternatively, you can request a temporary storage extension through our admin portal.</p>
<p>To manage your storage allocation, please log into the admin console:</p>
<p><a href="http://secure-nebula.com/admin/storage-mgmt?org=luxconsultancy">Manage Storage Settings</a></p>
<p>There is no immediate deadline, but we advise taking action within the next two weeks to avoid disruption to your team's workflows.</p>
<p>Best regards,<br>SecureNebula Cloud Services<br>Customer Support Team</p>
</div>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "Marco Bellini", "se": "m.bellini@euroconference-solutions.com",
     "su": "Complimentary VIP Pass \u2014 Luxembourg Admin Professionals Summit (Respond Today)",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background:#2c5f2d;padding:18px 20px;">
<h2 style="color:#fff;margin:0;font-size:18px;">Luxembourg Admin Professionals Summit 2026</h2>
<p style="color:#d4edda;margin:4px 0 0 0;font-size:13px;">16\u201317 April \u2022 Hotel Le Royal, Luxembourg City</p>
</div>
<div style="padding:20px;border:1px solid #ddd;border-top:none;">
<p>Dear Executive Support Professional,</p>
<p>On behalf of the organising committee, I am delighted to offer you a <strong>complimentary VIP delegate pass</strong> (valued at \u20ac490) for the upcoming Luxembourg Admin Professionals Summit.</p>
<p>The VIP pass includes:</p>
<ul>
<li>Priority seating at all keynote sessions</li>
<li>Exclusive networking dinner on 16 April</li>
<li>Masterclass: "AI Tools for Executive Assistants"</li>
<li>Certificate of Professional Development (8 CPD hours)</li>
</ul>
<p><strong>Only 5 complimentary passes remain.</strong> To secure yours, please register with your professional details before midnight tonight:</p>
<p style="text-align:center;margin:20px 0;">
<a href="http://euroconference-solutions.com/lux-admin-summit/vip-register" style="background-color:#2c5f2d;color:#fff;padding:12px 28px;text-decoration:none;border-radius:4px;display:inline-block;font-weight:bold;">Claim Your VIP Pass</a>
</p>
<p>Kind regards,<br>Marco Bellini<br>Event Director<br>EuroConference Solutions</p>
</div></div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "Sophie Kieffer", "se": "sophie.kieffer@luxconsultancy.com",
     "su": "New Perk: Discounted Corporate Gym Memberships",
     "bo": """<p>Hi everyone,</p>
<p>I'm happy to share that we've partnered with Fitness Loft Luxembourg to offer all LuxConsultancy employees a <strong>30% discount</strong> on annual gym memberships. This is part of our ongoing employee well-being initiative.</p>
<p>A few details:</p>
<ul>
<li>Discount applies to all Fitness Loft locations in Luxembourg City and Kirchberg</li>
<li>You can sign up anytime between now and 30 June 2026</li>
<li>Simply show your LuxConsultancy employee badge at reception</li>
</ul>
<p>If any of the partners or directors you support are interested, feel free to pass this along. There's no rush \u2014 the offer is valid for the full quarter.</p>
<p>Let me know if you have questions!</p>
<p>Best,<br>Sophie Kieffer<br>Office Manager<br>LuxConsultancy<br>sophie.kieffer@luxconsultancy.com</p>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "Luxembourg Fire Safety Authority", "se": "inspections@cspi.lu",
     "su": "URGENT: Mandatory Fire Drill Scheduled for Your Building \u2014 31 March",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="border-left:4px solid #cc0000;padding-left:15px;margin-bottom:15px;">
<h3 style="color:#cc0000;margin:0;">Mandatory Fire Safety Drill Notice</h3>
</div>
<p>Dear Building Tenant,</p>
<p>In accordance with Luxembourg fire safety regulation RGD 2026-01-15, a <strong>mandatory fire evacuation drill</strong> will be conducted at your office building on <strong>Monday, 31 March 2026 at 10:00</strong>.</p>
<p>As the administrative contact for LuxConsultancy, please ensure the following:</p>
<ul>
<li>All employees on your floor are informed of the drill</li>
<li>Designated fire wardens are briefed and have their high-visibility vests ready</li>
<li>Visitors present during the drill are escorted to the assembly point</li>
<li>Any mobility-impaired personnel are identified to reception before 09:30</li>
</ul>
<p><strong>Non-compliance may result in a formal citation and a fine of up to \u20ac2,500 per violation under Article 34 of the Fire Safety Code.</strong></p>
<p>Please confirm receipt of this notice by replying to this email. For questions, contact our inspections office at +352 247-84600.</p>
<p>Regards,<br>Service Incendie et Secours<br>Corps Grand-Ducal d'Incendie et de Secours (CGDIS)<br><a href="https://cspi.lu">https://cspi.lu</a></p>
</div>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "Thomas Weber", "se": "thomas.weber@lockgrid.com",
     "su": "Critical: Office Alarm System Maintenance Tonight \u2014 Codes Reset Required",
     "bo": """<p>Hi,</p>
<p>This is a reminder that LockGrid will be performing <strong>scheduled maintenance on LuxConsultancy's alarm system tonight between 22:00 and 02:00</strong>. During this window, all zone access codes will be temporarily reset.</p>
<p><strong>What you need to do before 18:00 today:</strong></p>
<ul>
<li>Ensure all office doors are secured manually before leaving</li>
<li>Inform any staff working late that the alarm panel will be offline</li>
<li>Do <strong>not</strong> attempt to arm or disarm the system during the maintenance window \u2014 doing so may trigger a false alarm and result in a police callout fee (\u20ac350)</li>
</ul>
<p>New access codes will be issued via our secure portal tomorrow morning. You will receive a separate email with login instructions from our regular address.</p>
<p>If you have any concerns or need to report an issue during maintenance, call our 24/7 line: <strong>+352 26 44 11 00</strong>.</p>
<p>Thank you for your cooperation,<br>Thomas Weber<br>Technical Account Manager<br>LockGrid Security Solutions<br><a href="https://lockgrid.com">www.lockgrid.com</a></p>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "Ana\u00efs Dumont", "se": "anais.dumont@papeterie-bernard.lu",
     "su": "New Corporate Catalogue Available \u2014 Special First-Order Discount",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
<div style="background-color:#4a90d9;padding:15px 20px;">
<h2 style="color:#ffffff;margin:0;font-size:17px;">Papeterie Bernard \u2014 Office Supplies Catalogue 2026</h2>
</div>
<div style="padding:20px;border:1px solid #e0e0e0;border-top:none;">
<p>Dear Office Manager,</p>
<p>I am writing to introduce Papeterie Bernard, a Luxembourg-based office supplies provider. We recently began servicing consulting firms in the Kirchberg area and would love the opportunity to work with LuxConsultancy.</p>
<p>Our 2026 corporate catalogue is now available, featuring:</p>
<ul>
<li>Eco-friendly stationery and printer supplies</li>
<li>Ergonomic desk accessories</li>
<li>Breakroom essentials with next-day delivery</li>
</ul>
<p>As a welcome offer, we are extending a <strong>15% discount on your first order</strong> \u2014 no minimum spend required. The offer is valid through the end of Q2.</p>
<p>You can browse the full catalogue on our website: <a href="https://papeterie-bernard.lu/corporate">https://papeterie-bernard.lu/corporate</a></p>
<p>Kind regards,<br>Ana\u00efs Dumont<br>Corporate Sales Representative<br>Papeterie Bernard S.\u00e0r.l.<br>+352 27 86 53 10<br><a href="https://papeterie-bernard.lu">papeterie-bernard.lu</a></p>
</div></div>"""},
]

COMPLIANCE_EMAILS = [
    # Cell 9: PHISHING, known sender, high urgency, reward framing, high quality
    {"sn": "LuxConsultancy Compliance Office", "se": "compliance-rewards@luxconsultancy.net",
     "su": "URGENT: You've Been Selected for the \u20ac2,500 Regulatory Excellence Award",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1b5e20;color:white;padding:15px;"><h2 style="margin:0;">Regulatory Excellence Recognition Program</h2></div>
<div style="padding:20px;">
<p>Dear Compliance Team Member,</p>
<p>Following the successful completion of our 2024 regulatory audit cycle, senior leadership has selected your team for the <strong>Annual Regulatory Excellence Award</strong>.</p>
<div style="background:#e8f5e9;padding:15px;text-align:center;margin:15px 0;border-radius:8px;">
<span style="font-size:32px;font-weight:bold;color:#2e7d32;">\u20ac2,500 Performance Bonus</span><br>
<span style="font-size:14px;color:#555;">Per eligible team member</span></div>
<p><strong style="color:#d32f2f;">You must confirm your acceptance within 6 hours to be included in this quarter's payout cycle.</strong></p>
<p style="text-align:center;"><a href="http://luxconsultancy-rewards.net/compliance-bonus/confirm" style="background:#2e7d32;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Accept Your Award</a></p>
<p style="font-size:12px;color:#666;">You will need to verify your employee ID and banking details for direct deposit. Ref: CRA-2025-Q1-0487</p>
</div></div>"""},
    # Cell 10: PHISHING, unknown sender, low urgency, threat framing, medium quality
    {"sn": "RegWatch Compliance Alerts", "se": "alerts@regwatch-compliance.net",
     "su": "GDPR Enforcement Trend Report - Potential Gaps Identified",
     "bo": """<p>Dear Compliance Officer,</p>
<p>RegWatch has completed its annual cross-industry GDPR compliance benchmarking study. Based on publicly available data, we have identified potential compliance gaps that may affect consulting firms operating in the EU, including LuxConsultancy.</p>
<p><strong>Areas of concern:</strong></p>
<ul><li>Data retention policies exceeding regulatory thresholds</li><li>Incomplete Records of Processing Activities (RoPA)</li><li>Third-party processor agreements lacking updated SCCs</li></ul>
<p>We recommend reviewing our detailed gap analysis report to assess whether these findings apply to your organization:</p>
<p><a href="http://regwatch-compliance.net/reports/gdpr-gaps-consulting-2025" style="color:#0078d4;">Download Gap Analysis Report (PDF)</a></p>
<p>While no immediate regulatory action is expected, firms that proactively address these areas tend to fare better during supervisory audits.</p>
<p>Kind regards,<br>RegWatch Compliance Alerts<br>European Regulatory Intelligence</p>"""},
    # Cell 11: PHISHING, known sender, low urgency, threat framing, medium quality
    {"sn": "GreenEnvi Compliance Team", "se": "compliance@greenenvi.org",
     "su": "Updated Environmental Due Diligence Questionnaire - Please Complete",
     "bo": """<p>Dear LuxConsultancy Compliance Team,</p>
<p>As part of our annual vendor risk assessment cycle, we are requesting all consulting partners to complete an updated environmental due diligence questionnaire.</p>
<p>This is a routine exercise, but please note that failure to complete the questionnaire could affect your vendor risk rating in our next review cycle.</p>
<p><strong>Questionnaire sections:</strong></p>
<ul><li>Environmental management system documentation</li><li>Carbon footprint reporting and targets</li><li>Supply chain sustainability practices</li><li>Regulatory compliance certifications (ISO 14001, etc.)</li></ul>
<p>Please access and complete the questionnaire here:</p>
<p><a href="http://greenenvi-portal.org/vendor-due-diligence/luxconsultancy" style="color:#0078d4;text-decoration:underline;">Complete Due Diligence Questionnaire</a></p>
<p>The suggested deadline is end of next month. Let us know if you need an extension.</p>
<p>Thank you,<br>GreenEnvi Compliance Team<br>compliance@greenenvi.org</p>"""},
    # Cell 12: PHISHING, unknown sender, high urgency, reward framing, high quality
    {"sn": "EU Anti-Money Laundering Institute", "se": "certifications@eu-amli.org",
     "su": "URGENT: Complimentary AMLA Certification - Registration Closes Tonight",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:linear-gradient(135deg,#0d47a1,#1565c0);color:white;padding:20px;text-align:center;">
<h2 style="margin:0;">EU Anti-Money Laundering Institute</h2>
<p style="margin:5px 0 0 0;font-size:13px;">Certified AML Professional Program</p></div>
<div style="padding:20px;">
<p>Dear LuxConsultancy Risk &amp; Compliance Team,</p>
<p>In preparation for the EU's new Anti-Money Laundering Authority (AMLA) framework taking effect in 2025, the EU-AMLI is offering <strong>fully sponsored AMLA certification</strong> to compliance professionals at select consulting firms.</p>
<div style="background:#e3f2fd;padding:15px;text-align:center;margin:15px 0;border:2px solid #1565c0;border-radius:8px;">
<p style="margin:0;font-size:14px;color:#555;">Certification Value</p>
<span style="font-size:28px;font-weight:bold;color:#0d47a1;">\u20ac4,800 \u2014 Fully Sponsored</span><br>
<span style="font-size:13px;color:#666;">Includes exam fees, study materials, and 2-day workshop in Brussels</span></div>
<p style="color:#d32f2f;"><strong>Only 5 seats remain. Registration closes tonight at 23:59 CET.</strong></p>
<p style="text-align:center;"><a href="http://eu-amli.org/certifications/amla-sponsored/register" style="background:#0d47a1;color:white !important;padding:14px 30px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Secure Your Seat Now</a></p>
<p style="font-size:12px;color:#666;">Professional credentials and employer verification required during registration.</p>
</div></div>"""},
    # Cell 13: LEGITIMATE, known sender, low urgency, reward framing
    {"sn": "Marie Fontaine (SuperFinance)", "se": "m.fontaine@superfinance.com",
     "su": "New AML/CFT Compliance Toolkit Available for Partners",
     "bo": """<p>Hi team,</p>
<p>Hope you're all doing well. I wanted to let you know that we've just released our updated <strong>AML/CFT Compliance Toolkit for 2025</strong>, and as a partner firm, LuxConsultancy gets complimentary access.</p>
<p>The toolkit includes:</p>
<ul><li>Updated customer due diligence (CDD) templates aligned with AMLD6</li><li>Risk scoring matrix for PEP and high-risk jurisdiction screening</li><li>Suspicious transaction reporting (STR) workflow guide</li><li>Sample policies for the new EU AMLA framework</li></ul>
<p>You can download everything from our partner resource hub: <a href="https://partners.superfinance.com/resources/aml-toolkit-2025" style="color:#0078d4;text-decoration:none;">AML/CFT Toolkit 2025</a></p>
<p>No rush on this \u2014 just a useful resource to have when you're updating your compliance documentation. Happy to walk through any of it on our next quarterly call.</p>
<p>Best regards,<br>Marie Fontaine<br>Regulatory Advisory, SuperFinance<br>+352 27 86 41 15</p>"""},
    # Cell 14: LEGITIMATE, unknown sender, high urgency, threat framing
    {"sn": "CSSF Supervisory Division", "se": "supervision@cssf-notifications.lu",
     "su": "URGENT: DORA Compliance Gap Assessment - Response Required Within 7 Days",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#1a237e;color:white;padding:15px;"><h2 style="margin:0;">Commission de Surveillance du Secteur Financier</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Supervisory Division \u2014 ICT Risk Oversight</p></div>
<div style="padding:20px;">
<p><strong>Reference:</strong> CSSF/DORA/2025/LUX-0293</p>
<p>Dear LuxConsultancy Compliance Officer,</p>
<p>As part of the Digital Operational Resilience Act (DORA) supervisory review cycle, your firm has been selected for a targeted compliance gap assessment. The following documentation must be submitted within <strong>7 business days</strong>:</p>
<ul><li>ICT risk management framework documentation</li><li>Third-party ICT service provider register (Article 28)</li><li>Digital operational resilience testing results (Article 26)</li><li>ICT-related incident reporting procedures</li></ul>
<p><strong style="color:#d32f2f;">Non-submission within the specified timeframe may result in a formal information request under Article 50 of the CSSF Law.</strong></p>
<p>Submit documentation via our secure portal: <a href="https://portal.cssf-notifications.lu/regulated-entities/dora-assessment" style="color:#0078d4;text-decoration:none;">CSSF Submission Portal</a></p>
<p>For procedural questions, contact our helpdesk at +352 26 251-2800.</p>
<p>Regards,<br>CSSF Supervisory Division<br>283, route d'Arlon, L-1150 Luxembourg</p>
</div></div>"""},
    # Cell 15: LEGITIMATE, known sender, high urgency, threat framing
    {"sn": "David M\u00fcller (LockGrid)", "se": "d.muller@lockgrid.com",
     "su": "URGENT: Critical Vulnerability in Compliance Monitoring Platform - Patch Required by Friday",
     "bo": """<p>Hi Compliance Team,</p>
<p>We've identified a <strong>critical vulnerability (CVE-2025-2847)</strong> in the compliance monitoring module that LuxConsultancy uses for automated regulatory reporting. This affects the data export function and could potentially expose sensitive audit trail data.</p>
<p><strong>Impact assessment:</strong></p>
<ol><li>Audit log export API vulnerable to unauthorized data extraction</li><li>Regulatory filing timestamps could be manipulated</li><li>Affects all clients on version 4.2.x of the compliance module</li></ol>
<p><strong style="color:#d32f2f;">We strongly recommend applying the emergency patch before end of business Friday. Unpatched systems will be flagged in your next regulatory audit report.</strong></p>
<p>Patch instructions and release notes are available here: <a href="https://portal.lockgrid.com/security/CVE-2025-2847/patch-instructions" style="color:#0078d4;text-decoration:none;">View Patch Instructions</a></p>
<p>Our team is available for assisted deployment if needed \u2014 just reply to this email or call me directly.</p>
<p>Best regards,<br>David M\u00fcller<br>Security Operations Lead<br>LockGrid Solutions<br>Direct: +31 26 48 71 45</p>"""},
    # Cell 16: LEGITIMATE, unknown sender, low urgency, reward framing
    {"sn": "European Compliance Forum", "se": "events@eurocomplianceforum.eu",
     "su": "Complimentary Access - Annual GRC Innovation Summit 2025",
     "bo": """<div style="font-family:Arial,sans-serif;max-width:600px;">
<div style="background:#37474f;color:white;padding:15px;"><h2 style="margin:0;">European Compliance Forum</h2>
<p style="margin:5px 0 0 0;font-size:12px;">Annual GRC Innovation Summit 2025</p></div>
<div style="padding:20px;">
<p>Dear Compliance Professional,</p>
<p>The European Compliance Forum is pleased to offer LuxConsultancy <strong>complimentary registration</strong> to our upcoming Governance, Risk &amp; Compliance Innovation Summit.</p>
<div style="background:#eceff1;padding:15px;margin:15px 0;border-left:4px solid #37474f;">
<p style="margin:0;"><strong>Date:</strong> May 14-15, 2025</p>
<p style="margin:5px 0;"><strong>Venue:</strong> European Convention Center Luxembourg (ECCL)</p>
<p style="margin:0;"><strong>Cost:</strong> <span style="color:#2e7d32;font-weight:bold;">FREE</span> for invited firms</p></div>
<p><strong>Featured sessions:</strong></p>
<ul><li>DORA implementation \u2014 lessons from early adopters</li><li>AI-driven compliance monitoring: opportunities and risks</li><li>Cross-border AML cooperation under the new AMLA framework</li><li>ESG regulatory landscape: what compliance teams need to know</li></ul>
<p style="text-align:center;"><a href="https://eurocomplianceforum.eu/events/grc-summit-2025/register" style="background:#37474f;color:white !important;padding:12px 25px;text-decoration:none !important;border-radius:4px;font-weight:bold;display:inline-block;">Register Now</a></p>
<p style="font-size:12px;color:#666;">Limited to 3 representatives per organization. CPE/CPD credits available.</p>
</div></div>"""},
]


# ============================================================
# SEED FUNCTION
# ============================================================

async def seed():
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    print("Clearing existing emails and participants...")
    await db.emails.delete_many({})
    await db.participants.delete_many({})

    all_emails = [WELCOME_EMAIL] + GENERIC_EMAILS

    clusters = {
        "Finance/Accounts Payable": FINANCE_EMAILS,
        "IT Support/Helpdesk": IT_EMAILS,
        "HR/People Operations": HR_EMAILS,
        "Sales/Business Development": SALES_EMAILS,
        "Operations/Logistics": OPS_EMAILS,
        "Customer Service/Client Support": CUSTOMER_SERVICE_EMAILS,
        "Marketing/Communications": MARKETING_EMAILS,
        "Procurement/Purchasing": PROCUREMENT_EMAILS,
        "Administrative/Executive Support": ADMIN_EMAILS,
        "Compliance/Risk/Audit": COMPLIANCE_EMAILS,
    }

    for cluster_name, content_list in clusters.items():
        if content_list:
            all_emails.extend(build_cluster_emails(cluster_name, content_list))

    result = await db.emails.insert_many(all_emails)
    print(f"Successfully inserted {len(result.inserted_ids)} emails")
    print(f"  - 1 welcome email")
    print(f"  - 8 generic emails (cells 1-8)")
    print(f"  - {len(result.inserted_ids) - 9} role-matched emails (cells 9-16 x 10 clusters)")
    print(f"\nTotal: {len(result.inserted_ids)} emails")

    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
