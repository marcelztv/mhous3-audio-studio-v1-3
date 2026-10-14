; Inno Setup script - produces installer_output\MHous3-Audio-Studio-Setup.exe
#define AppName "MHous3 Audio Studio"
#define AppVer "1.3.0"
[Setup]
AppId={{6F1B7E52-3C9A-4D0E-9B41-5A2D8C7E1F03}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher=M Hous3 Media
AppCopyright=Copyright (c) M Hous3 Media. All rights reserved.
VersionInfoCompany=M Hous3 Media
VersionInfoProductName=MHous3 Audio Studio
VersionInfoVersion={#AppVer}.0
LicenseFile=..\LICENSE.txt
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppName}.exe
OutputDir=..\installer_output
OutputBaseFilename=MHous3-Audio-Studio-Setup
SetupIconFile=..\assets\icon.ico
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
[Files]
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppName}.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppName}.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\{#AppName}.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
