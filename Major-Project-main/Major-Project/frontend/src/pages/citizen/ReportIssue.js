import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import {
  Camera, Video, Mic, MicOff, MapPin, Cpu, CheckCircle2, ChevronLeft, ChevronRight,
  ImagePlus, Loader2, X, Sparkles, AlertTriangle, Edit3, Navigation, PartyPopper,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DEPARTMENTS, DEPARTMENT_META, PRIORITY_META } from '@/lib/constants';
import { ConfidenceMeter } from '@/components/common/ConfidenceMeter';
import { PriorityBadge, DepartmentChip } from '@/components/common/Badges';
import { StylizedCityMap } from '@/components/common/StylizedCityMap';
import { complaintApi } from '@/lib/api';
import { fileToCompressedBase64 } from '@/lib/format';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

const PRIORITIES = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
const STEP_KEYS = ['step_evidence', 'step_describe', 'step_location', 'step_analysis', 'step_confirm'];
const isStockImageDescription = (value) => /^(an image has been uploaded|the image could not be classified confidently)/i.test((value || '').trim());
const GEOCODE_MIN_INTERVAL_MS = 1100;

export default function ReportIssue() {
  const navigate = useNavigate();
  const STEPS = STEP_KEYS.map((key) => t(key));
  const [step, setStep] = useState(0);
  const [image, setImage] = useState(null); // base64
  const [video, setVideo] = useState(null); // {name,url}
  const [description, setDescription] = useState('');
  const [listening, setListening] = useState(false);
  const [describingImage, setDescribingImage] = useState(false);
  const [location, setLocation] = useState(null); // {lat,lng,address,ward}
  const [locationLookup, setLocationLookup] = useState('');
  const [locationError, setLocationError] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [ai, setAi] = useState(null);
  const [edited, setEdited] = useState({ title: '', department: '', priority: '' });
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(null);
  const recRef = useRef(null);
  const photoInput = useRef(null);
  const videoInput = useRef(null);
  const geocodeController = useRef(null);
  const lastGeocodeAt = useRef(0);

  useEffect(() => () => geocodeController.current?.abort(), []);

  const speechSupported = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition);

  const describeImageFromUpload = async (base64) => {
    if (!base64) return;
    setDescribingImage(true);
    try {
      const res = await complaintApi.describeImage(base64);
      if (res?.description) {
        setDescription((current) => current.trim() && !isStockImageDescription(current) ? current : res.description);
      }
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Image description failed');
    } finally {
      setDescribingImage(false);
    }
  };

  const handlePhoto = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const b64 = await fileToCompressedBase64(file, 768, 0.68);
      setImage(b64);
    } catch { toast.error('Could not read image'); }
  };
  const handleVideo = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideo({ name: file.name, url: URL.createObjectURL(file) });
  };

  const toggleVoice = () => {
    if (!speechSupported) { toast.info('Voice input is not supported in this browser'); return; }
    if (listening) { recRef.current?.stop(); setListening(false); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.lang = 'en-IN'; rec.interimResults = false; rec.continuous = false;
    rec.onresult = (ev) => {
      const text = Array.from(ev.results).map((r) => r[0].transcript).join(' ');
      setDescription((d) => (d ? d + ' ' : '') + text);
    };
    rec.onerror = () => { toast.error('Voice capture failed'); setListening(false); };
    rec.onend = () => setListening(false);
    recRef.current = rec; rec.start(); setListening(true);
  };

  const selectLocation = async ({ lat, lng }) => {
    geocodeController.current?.abort();
    const controller = new AbortController();
    geocodeController.current = controller;
    setLocation({ lat, lng, ward: '', address: '' });
    setLocationError('');
    setLocationLookup('Finding address and area…');

    const wait = Math.max(0, GEOCODE_MIN_INTERVAL_MS - (Date.now() - lastGeocodeAt.current));
    if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
    if (controller.signal.aborted) return;
    lastGeocodeAt.current = Date.now();

    try {
      const params = new URLSearchParams({
        format: 'jsonv2',
        lat: String(lat),
        lon: String(lng),
        zoom: '18',
        addressdetails: '1',
      });
      const response = await fetch(`https://nominatim.openstreetmap.org/reverse?${params}`, {
        headers: { 'Accept-Language': 'en' },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Address lookup failed (${response.status})`);
      const result = await response.json();
      if (result.error) throw new Error(result.error);

      const address = result.address || {};
      const area = address.ward || address.suburb || address.neighbourhood || address.city_district
        || address.borough || address.quarter || address.city || address.town || '';
      if (geocodeController.current !== controller) return;
      setLocation((current) => current ? {
        ...current,
        ward: area,
        address: result.display_name || '',
      } : current);
      if (!area || !result.display_name) {
        setLocationError('Map details are incomplete. Fill in the missing area or address.');
      }
    } catch (error) {
      if (controller.signal.aborted || geocodeController.current !== controller) return;
      setLocationError('Could not look up this point. Enter the area and address manually.');
    } finally {
      if (geocodeController.current === controller) setLocationLookup('');
    }
  };

  const useCurrentLocation = () => {
    if (!navigator.geolocation) { toast.info('Location is unavailable — choose a point on the map'); return; }
    toast.loading('Getting your location…', { id: 'geo' });
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude: lat, longitude: lng } = pos.coords;
        void selectLocation({ lat, lng });
        toast.success('Location captured. Looking up the address…', { id: 'geo' });
      },
      () => {
        toast.error('Location denied — choose a point on the map', { id: 'geo' });
      },
      { enableHighAccuracy: true, timeout: 8000 },
    );
  };

  const runAnalysis = async () => {
    if (analyzing) return;
    setAnalyzing(true);
    setAi(null);
    try {
      const res = await complaintApi.analyze({ description, image_base64: image, location });
      if (res) {
        setAi(res);
        const titleToSet = res.detected_issue || res.category || (description ? description.slice(0, 60) : 'Reported Civic Issue');
        setEdited({
          title: titleToSet,
          department: res.department || 'Roads & Infrastructure',
          priority: res.priority || 'MEDIUM',
        });
      }
    } catch (err) {
      const detail = err?.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : 'AI analysis failed — please try again');
    } finally {
      setAnalyzing(false);
    }
  };

  const goNext = () => {
    if (step === 2) {
      runAnalysis();
    }
    setStep((s) => s + 1);
  };

  const canNext = useMemo(() => {
    if (step === 0) return true; // evidence optional but encouraged
    if (step === 1) return description.trim().length >= 3 || !!image;
    if (step === 2) return !!location && !!location.ward.trim() && !!location.address.trim() && !locationLookup;
    if (step === 3) return !analyzing && !!ai && !!edited.department && !!edited.priority;
    return true;
  }, [step, description, image, location, locationLookup, ai, edited, analyzing]);

  const submit = async () => {
    setSubmitting(true);
    try {
      const media = [];
      if (image) media.push({ type: 'image', data: image, name: 'photo.jpg' });
      if (video) media.push({ type: 'video', name: video.name });
      if (listening) recRef.current?.stop();
      const payload = {
        title: edited.title, description,
        department: edited.department, category: ai?.category || edited.department.split(' ')[0],
        priority: edited.priority, location, media,
        ai_prediction: { ...ai, department: edited.department, priority: edited.priority, detected_issue: edited.title },
      };
      const created = await complaintApi.create(payload);
      setDone(created);
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Could not submit complaint');
    } finally {
      setSubmitting(false);
    }
  };

  if (done) {
    return (
      <div className="flex min-h-[70vh] flex-col items-center justify-center px-6 text-center">
        <div className="grid h-20 w-20 place-items-center rounded-full bg-success/12 text-success">
          <PartyPopper className="h-10 w-10" />
        </div>
        <h1 className="mt-5 font-display text-2xl font-bold text-foreground">Complaint submitted!</h1>
        <p className="mt-2 max-w-sm text-sm text-muted-foreground">
          Your report has been routed to <span className="font-semibold text-foreground">{done.department}</span>. Track its progress anytime.
        </p>
        <div className="mt-4 rounded-xl border border-border bg-card px-5 py-3">
          <div className="text-xs text-muted-foreground">Tracking ID</div>
          <div className="font-mono text-lg font-bold text-primary">{done.tracking_id}</div>
        </div>
        <div className="mt-7 flex w-full max-w-xs flex-col gap-2">
          <Button onClick={() => navigate(`/app/complaints/${done.id}`)}>Track complaint</Button>
          <Button variant="outline" onClick={() => navigate('/app')}>Back to home</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-5">
      {/* Stepper */}
      <div className="mb-5 flex items-center gap-1.5">
        <button onClick={() => (step === 0 ? navigate('/app') : setStep((s) => s - 1))} className="grid h-9 w-9 place-items-center rounded-lg border border-border">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div className="flex flex-1 items-center gap-1.5">
          {STEPS.map((s, i) => (
            <div key={s} className={cn('h-1.5 flex-1 rounded-full transition-colors', i <= step ? 'bg-primary' : 'bg-secondary')} />
          ))}
        </div>
      </div>
      <div className="mb-4">
        <div className="text-xs font-semibold uppercase tracking-wide text-primary">Step {step + 1} of {STEPS.length}</div>
        <h1 className="font-display text-2xl font-bold text-foreground">{STEPS[step]}</h1>
      </div>

      {/* STEP 0: Evidence */}
      {step === 0 && (
        <div className="space-y-4 animate-fade-in">
          <p className="text-sm text-muted-foreground">Add a photo or short video of the issue. A clear photo helps the AI classify it accurately.</p>
          <input ref={photoInput} type="file" accept="image/*" capture="environment" className="hidden" onChange={handlePhoto} />
          <input ref={videoInput} type="file" accept="video/*" capture="environment" className="hidden" onChange={handleVideo} />

          {image ? (
            <div className="relative overflow-hidden rounded-2xl border border-border">
              <img src={image} alt="evidence" className="h-64 w-full object-cover" />
              <button onClick={() => setImage(null)} className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-full bg-black/60 text-white"><X className="h-4 w-4" /></button>
            </div>
          ) : (
            <button onClick={() => photoInput.current?.click()} className="flex h-48 w-full flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-border bg-muted/40 text-muted-foreground transition-colors hover:border-primary/50 hover:text-primary">
              <Camera className="h-9 w-9" />
              <span className="text-sm font-medium">Take or upload a photo</span>
            </button>
          )}

          {image && (
            <Button variant="outline" onClick={() => describeImageFromUpload(image)} disabled={describingImage} className="w-full gap-2">
              {describingImage ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} 
              {describingImage ? 'Analyzing image…' : 'Describe image with AI'}
            </Button>
          )}

          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline" onClick={() => photoInput.current?.click()} className="gap-2"><ImagePlus className="h-4 w-4" /> {image ? 'Change' : 'Photo'}</Button>
            <Button variant="outline" onClick={() => videoInput.current?.click()} className="gap-2"><Video className="h-4 w-4" /> {video ? 'Change video' : 'Video'}</Button>
          </div>
          {video && (
            <div className="flex items-center justify-between rounded-xl border border-border bg-card px-3 py-2 text-sm">
              <span className="inline-flex items-center gap-2 truncate"><Video className="h-4 w-4 text-accent" /> {video.name}</span>
              <button onClick={() => setVideo(null)}><X className="h-4 w-4 text-muted-foreground" /></button>
            </div>
          )}
        </div>
      )}

      {/* STEP 1: Describe */}
      {step === 1 && (
        <div className="space-y-4 animate-fade-in">
          <p className="text-sm text-muted-foreground">Describe what is wrong. You can type or use your voice.</p>
          <div className="relative">
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={6}
              placeholder="e.g. There is a large water pipeline leak flooding the road near the bus stop…" className="resize-none pr-12" />
            <button onClick={toggleVoice}
              className={cn('absolute bottom-3 right-3 grid h-9 w-9 place-items-center rounded-full transition-colors',
                listening ? 'bg-destructive text-destructive-foreground' : 'bg-primary/10 text-primary hover:bg-primary/20')}>
              {listening ? <MicOff className="h-4.5 w-4.5" /> : <Mic className="h-4.5 w-4.5" />}
            </button>
          </div>
          {listening && (
            <div className="flex items-center gap-2 text-sm text-destructive">
              <span className="flex h-2 w-2"><span className="absolute h-2 w-2 animate-ping rounded-full bg-destructive/60" /><span className="h-2 w-2 rounded-full bg-destructive" /></span>
              Listening… speak now
            </div>
          )}
          <p className="text-xs text-muted-foreground">{speechSupported ? 'Tip: Tap the mic and speak in English.' : 'Voice input isn\'t available in this browser — typing works fine.'}</p>
        </div>
      )}

      {/* STEP 2: Location */}
      {step === 2 && (
        <div className="space-y-4 animate-fade-in">
          <p className="text-sm text-muted-foreground">Click the map to place the pin on the exact issue location. The address and area are looked up from that point.</p>
          <Button onClick={useCurrentLocation} variant="outline" className="w-full gap-2"><Navigation className="h-4 w-4" /> Use my current location</Button>
          <StylizedCityMap
            height={280}
            showHotspots={false}
            pickerHint="Click map to set exact location"
            onMapClick={(coords) => { void selectLocation(coords); }}
            points={location ? [{
              id: 'reported-location',
              title: description || 'Reported issue',
              priority: edited.priority || 'MEDIUM',
              department: 'Roads & Infrastructure',
              location,
            }] : []}
          />
          {locationLookup && <p className="text-sm text-muted-foreground">{locationLookup}</p>}
          <div className="space-y-1.5">
            <Label htmlFor="location-area">Ward / Area</Label>
            <Input
              id="location-area"
              value={location?.ward || ''}
              onChange={(e) => setLocation((current) => current ? { ...current, ward: e.target.value } : current)}
              placeholder={location ? 'Area from map or enter the ward' : 'Select a point on the map first'}
              disabled={!location || !!locationLookup}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="location-address">Exact address / landmark</Label>
            <Input
              id="location-address"
              value={location?.address || ''}
              onChange={(e) => setLocation((current) => current ? { ...current, address: e.target.value } : current)}
              placeholder={location ? 'Address from map or enter a nearby landmark' : 'Select a point on the map first'}
              disabled={!location || !!locationLookup}
            />
          </div>
          {location && (
            <p className="flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
              <MapPin className="h-3.5 w-3.5" /> {location.lat.toFixed(6)}, {location.lng.toFixed(6)}
            </p>
          )}
          {locationError && <p role="alert" className="text-sm text-destructive">{locationError}</p>}
        </div>
      )}

      {/* STEP 3: AI Analysis */}
      {step === 3 && (
        <div className="space-y-4 animate-fade-in">
          {analyzing || !ai ? (
            <div className="flex flex-col items-center py-8">
              <div className="relative overflow-hidden rounded-2xl border border-border">
                {image ? <img src={image} alt="analyzing" className="h-56 w-full max-w-xs object-cover opacity-90" />
                  : <div className="grid h-56 w-72 place-items-center bg-muted text-muted-foreground"><Cpu className="h-10 w-10" /></div>}
                <div className="absolute inset-x-0 top-0 h-1 bg-primary/70 shadow-glow" style={{ animation: 'scan 1.6s ease-in-out infinite' }} />
                <div className="absolute inset-0 bg-primary/5" />
              </div>
              <div className="mt-5 flex items-center gap-2 text-sm font-medium text-primary">
                <Loader2 className="h-4 w-4 animate-spin" /> AI is analysing your photo, description &amp; location…
              </div>
              {!analyzing && !ai && (
                <Button onClick={runAnalysis} className="mt-4 gap-2">
                  <Sparkles className="h-4 w-4" /> Try analysis again
                </Button>
              )}
            </div>
          ) : (
            <>
              <div className="rounded-2xl border border-primary/25 bg-primary/[0.04] p-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-primary">
                  <Sparkles className="h-4 w-4" /> AI analysed your report
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{ai.reasoning}</p>
              </div>

              {(edited.priority === 'CRITICAL' || ai.safety_flag) && (
                <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/8 p-3 text-sm text-destructive">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>Possible safety hazard detected. This will be flagged as high urgency.</span>
                </div>
              )}

              <div className="space-y-1.5">
                <Label className="flex items-center gap-1.5"><Edit3 className="h-3.5 w-3.5" /> Detected issue</Label>
                <Input value={edited.title || ai?.detected_issue || ai?.category || ''} onChange={(e) => setEdited({ ...edited, title: e.target.value })} placeholder="Detected civic issue..." />
              </div>

              <div className="grid grid-cols-1 gap-3">
                <div className="space-y-1.5">
                  <Label>Department <span className="font-normal text-muted-foreground">(correct if needed)</span></Label>
                  <Select value={edited.department} onValueChange={(v) => setEdited({ ...edited, department: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>{DEPARTMENTS.map((d) => <SelectItem key={d} value={d}>{d}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Priority</Label>
                  <Select value={edited.priority} onValueChange={(v) => setEdited({ ...edited, priority: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>{PRIORITIES.map((p) => <SelectItem key={p} value={p}>{PRIORITY_META[p].label}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-card p-4">
                <ConfidenceMeter value={ai.confidence} source={ai.source} />
                {ai.tags?.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {ai.tags.map((tag) => <span key={tag} className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium text-secondary-foreground">#{tag}</span>)}
                  </div>
                )}
              </div>
              <Button variant="ghost" onClick={runAnalysis} className="w-full gap-2 text-primary"><Cpu className="h-4 w-4" /> Re-run analysis</Button>
            </>
          )}
        </div>
      )}

      {/* STEP 4: Confirm */}
      {step === 4 && (
        <div className="space-y-4 animate-fade-in">
          <p className="text-sm text-muted-foreground">Review your report before submitting.</p>
          {image && <img src={image} alt="evidence" className="h-48 w-full rounded-2xl border border-border object-cover" />}
          <div className="space-y-3 rounded-2xl border border-border bg-card p-4">
            <Row label="Issue" value={edited.title} />
            <div className="flex items-center justify-between"><span className="text-sm text-muted-foreground">Department</span><DepartmentChip department={edited.department} short /></div>
            <div className="flex items-center justify-between"><span className="text-sm text-muted-foreground">Priority</span><PriorityBadge priority={edited.priority} /></div>
            <Row label="Location" value={location?.address} />
            {description && <Row label="Description" value={description} />}
          </div>
          <p className="text-center text-xs text-muted-foreground">By submitting, you confirm the information is accurate to the best of your knowledge.</p>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-7">
        {step < 4 ? (
          <Button className="w-full gap-2" disabled={!canNext} onClick={goNext}>
            {step === 3 ? 'Looks good, continue' : 'Continue'} <ChevronRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button className="w-full gap-2" disabled={submitting} onClick={submit}>
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />} Submit Complaint
          </Button>
        )}
        {step === 0 && !image && !video && (
          <button onClick={() => setStep(1)} className="mt-3 w-full text-center text-sm text-muted-foreground hover:text-foreground">Skip — continue without media</button>
        )}
      </div>
    </div>
  );
}

const Row = ({ label, value }) => (
  <div className="flex items-start justify-between gap-4">
    <span className="text-sm text-muted-foreground">{label}</span>
    <span className="max-w-[65%] text-right text-sm font-medium text-foreground">{value}</span>
  </div>
);
