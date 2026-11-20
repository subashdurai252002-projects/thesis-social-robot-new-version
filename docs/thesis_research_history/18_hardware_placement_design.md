# Camera, ReSpeaker and Speaker Placement Design

Date: 20 November 2026

## Purpose

This document defines the intended placement of the camera, ReSpeaker microphone array and loudspeaker for the laptop prototype and later physical-robot installation.

Correct placement is necessary for:

- Reliable face and gaze observation.
- Passive RGB liveness.
- ReSpeaker direction estimation.
- Visual and audio intended-speaker matching.
- Reduced loudspeaker echo.
- Repeatable thesis experiments.

This is a design document. Final measurements and mounting positions must be validated on the physical robot in the laboratory.

## Main design principle

The camera and ReSpeaker must use the same forward-facing reference.

```text
                    PERSON

                       ↑
                Robot forward axis
                       ↑

                [ Camera lens ]
                       |
                [ ReSpeaker ]
                       |
                Robot centre line

When a person appears on the camera’s left, their speech should also be classified on the corresponding audio side after coordinate calibration.

The system must not assume that the ReSpeaker’s physical left and right automatically match the camera image. A calibration experiment must establish the mapping after mounting.

## Coordinate system

Use one robot-relative coordinate system:

- Forward: directly in front of the robot.
- Left: robot’s left when facing forward.
- Right: robot’s right when facing forward.
- Up: toward the top of the robot.
- Down: toward the floor.

Important distinction:

- Camera-image left may be different from the person’s own left.
- ReSpeaker delay signs depend on microphone order and board orientation.
- ROS coordinates may use positive and negative angles differently.

The final implementation must store an explicit camera-to-audio direction mapping instead of relying on assumptions.

## Recommended camera placement

The camera should be:

- Near the robot’s visual centre line.
- Above or close to the ReSpeaker.
- Facing the same forward direction as the robot.
- Approximately level with users’ upper body or face where practical.
- Rigidly mounted so its angle does not move during an experiment.
- Free from obstruction by cables, robot parts or the microphone.
- Positioned to observe both people during multi-person experiments.

Avoid:

- Extreme upward or downward camera tilt.
- Mounting far to one side.
- Loose mounts.
- Strong backlighting directly behind participants.
- Placing the loudspeaker where it blocks the camera.

## Recommended ReSpeaker placement

The ReSpeaker should be:

- Close to the camera’s horizontal centre line.
- Mounted with a fixed and documented orientation.
- Facing upward or in the manufacturer-supported orientation.
- Uncovered and free from cloth, foam or robot panels.
- Isolated from direct mechanical vibration where possible.
- Kept away from cooling fans and motor noise.
- Mounted so all microphones remain exposed.

The board orientation must be marked physically after calibration:

CAMERA FORWARD
      ↑
      |
[ReSpeaker reference mark]

The reference mark should remain unchanged for all recordings in the same experiment.

Do not label the board’s sides as audio left or audio right until a controlled calibration confirms the mapping.

## Recommended loudspeaker placement

The loudspeaker should be:

- Fixed relative to the camera and ReSpeaker.
- Positioned so it does not block the camera or microphones.
- Separated from the ReSpeaker as much as the robot design allows.
- Directed toward the user rather than directly toward the microphones.
- Mechanically isolated to reduce vibration transfer.
- Used at the lowest volume that remains clearly understandable.

The speaker position must remain constant during comparable experiments.

Channel 5 should continue to provide the playback-reference signal for echo and barge-in processing.

## Preferred relative layout

Front view:

             [ CAMERA ]

         [ ReSpeaker array ]

        [ ROBOT / DISPLAY ]

     [LOUDSPEAKER]   or   [LOUDSPEAKERS]

Top view:

                         User area
                left       centre       right
                  \           |           /
                   \          |          /
                    \         |         /
                     [Camera direction]
                     [ReSpeaker centre]
                     [Robot centre line]

              Loudspeaker positioned away from
              direct microphone-array exposure

The exact speaker position may differ depending on the robot body, but it must be documented.

## Laptop prototype placement

The home laptop arrangement is a temporary prototype and not the final robot installation.

For repeatable laptop tests:

- Keep the laptop camera angle fixed.
- Place the ReSpeaker in front of the laptop.
- Align its centre approximately with the webcam centre line.
- Mark the ReSpeaker position on the table.
- Mark the laptop position and screen angle.
- Keep the external speaker in one fixed location.
- Avoid moving the laptop, ReSpeaker or speaker between repeated trials.
- Record any unavoidable placement change in the experiment metadata.

The existing home experiments remain valid as controlled prototype evidence, but their direction mapping must not automatically be transferred to the robot.

## Final robot-placement measurements

Record the following in the laboratory:

| Measurement | Required value |
|---|---|
| Camera height from floor | Pending |
| ReSpeaker height from floor | Pending |
| Speaker height from floor | Pending |
| Camera-to-ReSpeaker horizontal offset | Pending |
| Camera-to-ReSpeaker vertical offset | Pending |
| Camera pitch angle | Pending |
| Camera yaw angle | Pending |
| ReSpeaker board orientation | Pending |
| ReSpeaker-to-speaker distance | Pending |
| Speaker direction | Pending |
| Typical user distance | Pending |
| Tested horizontal user angles | Pending |

Photograph the completed installation from:

- Front.
- Left side.
- Right side.
- Top or elevated view where possible.

Do not include participant faces in public documentation without consent.

## Camera–audio alignment procedure

After mounting:

1. Place one speaker directly in front of the system.
2. Record camera position and ReSpeaker direction evidence.
3. Repeat at a known left position.
4. Repeat at a known right position.
5. Confirm which ReSpeaker delay sign corresponds to camera left and right.
6. Store the mapping in configuration.
7. Repeat the test after any hardware movement.
8. Reject direction matching when confidence is insufficient.

This calibration should use several repetitions and more than one speaker where possible.

## Intended-speaker matching

Future intended-speaker selection should follow:

Camera identifies the attention candidate
                +
Camera estimates the candidate’s horizontal position
                +
ReSpeaker estimates the active speech direction
                ↓
Compare visual and audio directions
                ↓
Accept, reject or mark uncertain

Suggested decision behaviour:

- Visual and audio directions agree: accept as supporting evidence.
- Directions clearly disagree: reject or suppress the speech turn.
- Audio direction is uncertain: do not claim speaker identity.
- Multiple people overlap: avoid forcing a decision without adequate evidence.

Audio direction should support intended-speaker selection. It should not replace visual tracking or independently control the robot.

## Robot orientation

If the intended person is located away from the robot’s centre, the robot may later orient toward that person.

However:

- Direction matching must be validated first.
- The interaction manager must approve the target.
- Movement must use the official Unitree interface.
- Initial tests must keep the robot stationary.
- Orientation speed and range must be limited.
- Manual supervision and emergency stop must be available.
- Ambiguous person identity must cancel orientation commands.

Physical robot movement is a later laboratory stage and is not part of the current home prototype.

## Placement risks and controls

| Risk | Control |
|---|---|
| Camera and audio left/right are reversed | Perform controlled alignment calibration |
| Hardware moves between trials | Mark and measure all positions |
| Speaker echo reaches microphones | Use playback reference and physical separation |
| Robot vibration contaminates audio | Add mechanical isolation and test while stationary first |
| Camera misses faces | Adjust height, tilt and field of view |
| ReSpeaker is obstructed | Keep all microphones exposed |
| Fan or motor noise affects speech | Increase separation and conduct noise tests |
| Multiple people share similar direction | Report uncertainty instead of forcing selection |
| Tracking ID changes during crossing | Do not command movement from uncertain identity |

## Validation conditions

The final placement should be tested with:

- One person at centre.
- One person at left.
- One person at right.
- Different distances.
- Two stationary people.
- People crossing positions.
- One intended speaker and one distractor.
- Alternating speech.
- Simultaneous speech.
- Robot loudspeaker playback.
- Background equipment noise.
- Different participant heights.

## Acceptance criteria

Placement is acceptable when:

- Both intended participants remain visible.
- Face and eye landmarks are reliably available.
- Screen or robot attention is accepted.
- Clear away conditions are rejected.
- Left, centre and right audio directions remain distinguishable.
- Camera and audio direction labels agree.
- Speech playback does not continually trigger false barge-in.
- Hardware remains fixed during repeated trials.
- The system reports uncertainty when evidence is insufficient.

## Current decision

The laptop setup remains a temporary prototype.

The final robot design should place the camera and ReSpeaker close to the same centre line, keep the speaker physically separated, and use an experimentally verified camera-to-audio mapping.

No physical robot movement or final direction mapping is claimed until the laboratory installation is measured and validated.
