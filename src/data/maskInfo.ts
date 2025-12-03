// 한국 전통 탈 정보 데이터베이스
export interface MaskDetailInfo {
  name: string;
  koreanName: string;
  origin: string;
  description: string;
  danceRole: string;
  quote: string;
  wikiUrl?: string;
}

export const MASK_INFO: Record<string, MaskDetailInfo> = {
  // 하회탈 (안동 하회마을) - 국보 제121호, 고려 중기 제작, 현존하는 가장 오래된 탈
  yangban: {
    name: 'yangban',
    koreanName: '양반',
    origin: '안동 하회탈',
    description: '하회탈 중 가장 유명한 탈로, 허례허식이 가득한 양반층을 상징합니다. 턱이 분리되어 얼굴을 들면 웃는 표정, 숙이면 성난 표정으로 변하는 것이 특징입니다. 파안대소하는 모습이 인상적입니다.',
    danceRole: '선비와 함께 풍자의 대상이 되며, 초랭이에게 조롱당하는 위선적 양반 역할',
    quote: '이놈, 상놈이 감히 양반을 희롱하느냐!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  bune: {
    name: 'bune',
    koreanName: '부네',
    origin: '안동 하회탈',
    description: '양반과 선비 풍자의 중심이 되는 술집 작부를 상징합니다. 당시 조선의 미인상을 보여주며, 한국 전통 탈 중 가장 아름다운 여성상으로 평가받습니다.',
    danceRole: '중의 유혹을 받으며 양반과 선비가 다투게 만드는 젊은 여인 역할',
    quote: '아이고, 저를 두고 싸우시다니요...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  gaksi: {
    name: 'gaksi',
    koreanName: '각시',
    origin: '안동 하회탈',
    description: '굿판에 쓰이는 여성 탈로, 사랑하는 이를 그리워하는 조선 여성들을 상징합니다. 전설에 따르면 허도령을 사모하다 죽은 김씨 처녀가 서낭신이 된 모습이라고도 합니다.',
    danceRole: '혼례 장면에서 순결하고 정숙한 새색시 역할',
    quote: '부끄러워라, 고개를 들 수가 없구나...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  jung: {
    name: 'jung',
    koreanName: '중',
    origin: '안동 하회탈',
    description: '불교의 가르침을 벗어난 파계승을 상징하는 탈입니다. 탐욕스럽고 음탕한 표정이 특징이며, 종교의 위선을 풍자합니다.',
    danceRole: '부네를 유혹하는 위선적인 파계승 역할',
    quote: '나무아미타불... 아니, 저 여인의 자태란!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  imae: {
    name: 'imae',
    koreanName: '이매',
    origin: '안동 하회탈',
    description: '유일하게 턱이 없는 탈로, 전설에 따르면 허도령이 탈을 완성하기 전에 죽었기 때문입니다. 어딘가 바보스럽거나 몸이 불편한 장애인을 상징하며, 탈춤 내 최고의 개그 캐릭터입니다.',
    danceRole: '어리숙하지만 양반의 허세를 본의 아니게 폭로하는 역할',
    quote: '에헤헤, 그게 뭔 소리여?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  chorangi: {
    name: 'chorangi',
    koreanName: '초랭이',
    origin: '안동 하회탈',
    description: '어딘가 비웃는 얼굴이 특징인 괴짜 캐릭터입니다. 양반의 일을 돕지만 양반과 선비를 놀리는 게 취미로, 탈춤 내 최고 인기 스타입니다. 백성을 대신해 양반층을 꾸짖는 주인공입니다.',
    danceRole: '양반을 수행하며 그의 허세를 꼬집는 영리한 하인 역할',
    quote: '나으리, 그건 좀 아닌 것 같은디요...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  baekjeong: {
    name: 'baekjeong',
    koreanName: '백정',
    origin: '안동 하회탈',
    description: '양반탈과 비슷하지만 투박하고 우악스러운 외모가 특징입니다. 천민 신분에 시달리는 도축업자인 백정의 원초적 강인함을 상징합니다.',
    danceRole: '소의 생식기를 들고 양반을 조롱하는 통쾌한 역할',
    quote: '양반 나으리, 이것 좀 보시오!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  halmi: {
    name: 'halmi',
    koreanName: '할미',
    origin: '안동 하회탈',
    description: '여성으로서의 매력이 사라졌지만 인생의 노련한 관록을 보여주는 할머니를 상징합니다. 주름진 얼굴과 이빨 빠진 입이 특징입니다.',
    danceRole: '억척스럽게 살아온 서민 여성의 애환을 표현하는 역할',
    quote: '이 늙은 것이 평생을 고생만 했구나...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  seonbi: {
    name: 'seonbi',
    koreanName: '선비',
    origin: '안동 하회탈',
    description: '양반과 함께 풍자의 대상이 되는 선비를 상징합니다. 학문에 정진하는 척하지만 실상은 허영심과 체면에 급급한 모습을 보여줍니다.',
    danceRole: '양반과 함께 부네를 두고 다투는 허세 가득한 지식인 역할',
    quote: '학문을 논하는 자리에 어찌 천한 것들이...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  juji: {
    name: 'juji',
    koreanName: '주지',
    origin: '안동 하회탈',
    description: '암컷과 수컷으로 구성된 손탈로, 사자를 형상화했습니다. 흥미롭게도 암컷 주지가 수컷보다 더 사납습니다. 벽사의 의미를 지닌 탈입니다.',
    danceRole: '탈놀이 시작 전 잡귀를 쫓는 벽사 의식 역할',
    quote: '(으르렁) 악귀를 쫓노라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  
  // 봉산탈춤 (황해도 봉산) - 국가무형유산 제17호, 해서탈춤의 대표
  malttugi: {
    name: 'malttugi',
    koreanName: '말뚝이',
    origin: '봉산탈춤',
    description: '봉산탈춤에서 가장 인기 있는 캐릭터로, 양반의 하인입니다. 채찍을 들고 등장하여 양반 삼형제를 신랄하게 비판하며 해학적이고 풍자적인 대사로 유명합니다.',
    danceRole: '양반들을 인도하고 등장하여 양반 사회의 부패와 비리를 해학으로 고발하는 역할',
    quote: '쉬이~ 양반 나오신다! 개다리소반이라는 반 자 쓰는 양반 나오신다!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  miyal: {
    name: 'miyal',
    koreanName: '미얄',
    origin: '봉산탈춤',
    description: '영감의 본처로, 원래 무당이었습니다. 난리통에 헤어졌다 만난 영감이 데려온 첩 때문에 싸움이 벌어지고, 결국 맞아 죽는 비극적 인물입니다. 서민 여성에 대한 부당한 횡포를 보여줍니다.',
    danceRole: '영감과 첩 덜머리집 사이에서 쫓겨나 죽는 비극적 역할',
    quote: '이 늙은 년을 버리고 젊은 년을 취하다니...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  chwibari: {
    name: 'chwibari',
    koreanName: '취발이',
    origin: '봉산탈춤',
    description: '노총각 역할을 하는 주요 인물로, 머리카락이 헝클어지고 술에 취한 듯한 붉은 얼굴이 특징입니다. 노장과 대결하여 승리하고 소무와 사랑을 나눕니다.',
    danceRole: '노장을 물리치고 소무와 사랑을 나눈 뒤 아이를 얻어 글을 가르치는 역할',
    quote: '아이고 취했다, 저 각시 좀 보소!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  nojang: {
    name: 'nojang',
    koreanName: '노장',
    origin: '봉산탈춤',
    description: '살아있는 부처라는 칭송을 받던 늙은 고승을 상징합니다. 도를 닦은 듯 하지만 결국 소무의 아름다움에 빠져 파계하는 위선적 모습을 보여줍니다.',
    danceRole: '소무에게 유혹되어 파계하고, 취발이에게 패하는 위선적 승려 역할',
    quote: '불도를 닦은 지 수십 년인데... 아, 저 여인이여!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  somu: {
    name: 'somu',
    koreanName: '소무',
    origin: '봉산탈춤',
    description: '노장, 취발이, 양반 등의 상대역으로 나오는 젊은 여자를 상징합니다. 아름답고 요염한 표정이 특징입니다.',
    danceRole: '노장을 유혹하여 파계하게 만드는 요염한 무희 역할',
    quote: '스님, 이리 오시어요~',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  yeongam: {
    name: 'yeongam',
    koreanName: '영감',
    origin: '봉산탈춤',
    description: '늙은 남편을 상징하는 탈로, 원래 땜쟁이였습니다. 첩인 돌머리집을 데려와 본처 미얄을 홀대하고 때려 죽이는 가부장적 횡포를 보여줍니다.',
    danceRole: '미얄할미와 덜머리집 사이에서 갈등하는 가부장 역할',
    quote: '에잇, 늙은 년은 물러가라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  meokjung: {
    name: 'meokjung',
    koreanName: '먹중',
    origin: '봉산탈춤',
    description: '팔먹중춤에 등장하는 여덟 명의 중입니다. 비사실적인 귀면형으로 요철 굴곡이 심한 것이 특징입니다. 각각 유식한 대사를 낭송한 뒤 활달한 춤을 춥니다.',
    danceRole: '봉산탈춤 중 가장 화려하고 남성적인 힘이 돋보이는 춤을 추는 역할',
    quote: '이런 좋은 풍류정을 만났으니 한바탕 놀고 가리라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  mokjung: {
    name: 'mokjung',
    koreanName: '목중',
    origin: '봉산탈춤',
    description: '팔먹중 중 다섯째 먹중으로, 목이 긴 형상이 특징입니다. 다른 먹중들과 함께 유식한 대사를 낭송하며 화려한 춤을 선보입니다.',
    danceRole: '팔먹중춤에서 다섯째로 등장하여 활달한 춤을 추는 역할',
    quote: '목이 길어 세상을 멀리 내다보노라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  omjung: {
    name: 'omjung',
    koreanName: '옴중',
    origin: '봉산탈춤',
    description: '팔먹중 중 셋째, 넷째 먹중으로, 온몸에 옴이 난 형상이 특징입니다. 귀면형의 탈로 요철 굴곡이 심하며 익살스러운 모습을 보여줍니다.',
    danceRole: '팔먹중춤에서 셋째, 넷째로 등장하여 익살스러운 춤을 추는 역할',
    quote: '온몸이 가려워도 춤은 멈출 수 없다!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  sangja: {
    name: 'sangja',
    koreanName: '상좌',
    origin: '봉산탈춤',
    description: '어린 중을 상징하며, 탈춤의 첫 번째 과장에서 등장합니다. 동서남북 사방신에게 놀이의 시작을 알리고 놀이판의 사악한 기운을 쫓습니다.',
    danceRole: '탈놀이 시작을 알리고 벽사의 의식무를 추는 역할',
    quote: '사방신이시여, 놀이판을 지켜주소서!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  
  // 고성오광대 (경남 고성)
  mundungi: {
    name: 'mundungi',
    koreanName: '문둥이',
    origin: '고성오광대',
    description: '문둥병 환자를 상징하는 탈로, 병으로 일그러진 얼굴이 특징입니다. 사회적 약자의 한을 표현하며, 오광대의 첫 번째 과장에서 등장합니다.',
    danceRole: '병자의 슬픔과 한을 춤으로 표현하는 역할',
    quote: '손가락이 다 떨어져 나가도 살아야 하는 이 목숨...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  bibi: {
    name: 'bibi',
    koreanName: '비비',
    origin: '고성오광대',
    description: '양반의 첩을 상징하는 탈로, 요염하고 교태로운 표정이 특징입니다.',
    danceRole: '양반과 말뚝이 사이 갈등의 원인이 되는 첩 역할',
    quote: '나으리, 저만 보시어요~',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  joje: {
    name: 'joje',
    koreanName: '조제',
    origin: '고성오광대',
    description: '제비를 상징하는 탈로, 흥부놀부전의 제비를 연상시키는 독특한 형태입니다.',
    danceRole: '제비의 날렵한 움직임을 춤으로 표현하는 역할',
    quote: '흥부야, 이 박씨를 심어라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  
  // 처용무 (궁중무용)
  cheoyong: {
    name: 'cheoyong',
    koreanName: '처용',
    origin: '처용무',
    description: '신라 헌강왕 때의 인물 처용을 형상화한 탈입니다. 역신을 물리치는 벽사의 의미를 가지며, 궁중무용인 처용무에 사용됩니다. 오방색 의상을 입고 춤을 춥니다.',
    danceRole: '오방색 의상을 입고 역신을 물리치는 궁중무용 역할',
    quote: '역신이여 물러가라! 내 아내를 범한 죄 용서하리니...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/처용무'
  },
  
  // 나례 (궁중 벽사 의식)
  bangsangsi: {
    name: 'bangsangsi',
    koreanName: '방상씨',
    origin: '나례',
    description: '귀신을 쫓는 벽사 의식에 사용되는 탈입니다. 네 개의 황금빛 눈이 특징이며, 곰가죽 옷을 입고 창과 방패를 들고 사방의 귀신을 쫓습니다.',
    danceRole: '궁중 나례 의식에서 역귀를 쫓는 벽사의 신 역할',
    quote: '악귀야 물러가라! 사방을 지키노라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/방상시'
  },
  
  // 송파산대놀이 (서울)
  songpa: {
    name: 'songpa',
    koreanName: '송파산대놀이탈',
    origin: '송파산대놀이',
    description: '서울 송파 지역의 산대놀이에 사용되는 탈입니다. 경기 지역 산대놀이의 영향을 받아 서민적이고 해학적인 표정이 특징입니다.',
    danceRole: '양반 풍자와 파계승 이야기를 펼치는 서울 지역 탈놀이 역할',
    quote: '양반인들 뭐가 다르단 말이오?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/송파산대놀이'
  },
  
  // 강령탈춤 (황해도)
  gangnyeong: {
    name: 'gangnyeong',
    koreanName: '강령탈',
    origin: '강령탈춤',
    description: '황해도 강령 지역의 탈춤에 사용되는 탈입니다. 봉산탈춤과 함께 해서탈춤의 쌍벽을 이루며, 사실적인 인물 가면으로 눈망울이 큰 것이 특징입니다.',
    danceRole: '역동적인 춤사위로 황해도를 대표하는 탈놀이 역할',
    quote: '에라, 모르겠다! 한바탕 놀아보세!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/강령탈춤'
  },
  
  // 통영오광대 (경남 통영)
  tongyeong: {
    name: 'tongyeong',
    koreanName: '통영오광대탈',
    origin: '통영오광대',
    description: '경남 통영 지역의 오광대놀이에 사용되는 탈입니다. 남해안 특유의 개방적이고 활달한 분위기가 느껴지며, 바다와 인접한 지역 특성이 반영되어 있습니다.',
    danceRole: '바다 인접 지역 특성의 활기차고 역동적인 탈놀이 역할',
    quote: '바다 사나이가 왔소! 한판 붙어볼까?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/통영오광대'
  },
  
  // 수영야류 (부산)
  suyeong: {
    name: 'suyeong',
    koreanName: '수영야류탈',
    origin: '수영야류',
    description: '부산 수영 지역의 야류(들놀이)에 사용되는 탈입니다. 영남 지역 특유의 강인한 느낌이 특징이며, 정월대보름에 주로 공연됩니다.',
    danceRole: '정월대보름 마을 축제를 이끄는 부산 대표 민속놀이 역할',
    quote: '영남 사나이 기개를 보여주마!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/수영야류'
  },
  
  // 동래야류 (부산 동래)
  dongnae: {
    name: 'dongnae',
    koreanName: '동래야류탈',
    origin: '동래야류',
    description: '부산 동래 지역의 야류에 사용되는 탈입니다. 수영야류와 함께 부산 지역의 대표적인 탈놀이로, 양반 풍자와 서민들의 해학이 담겨 있습니다.',
    danceRole: '동래 지역의 정월대보름 축제를 이끄는 탈놀이 역할',
    quote: '온천의 기운을 받아 신명나게 놀아보세!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/동래야류'
  },
  
  // 양주별산대놀이 (경기 양주)
  yangju: {
    name: 'yangju',
    koreanName: '양주별산대놀이탈',
    origin: '양주별산대놀이',
    description: '경기도 양주 지역의 산대놀이에 사용되는 탈입니다. 서울 본산대놀이의 영향을 받았으며, 지방 산대놀이의 대표격으로 국가무형유산입니다.',
    danceRole: '경기 지역 산대놀이의 전통을 잇는 탈놀이 역할',
    quote: '한양 가까운 양주 땅에서 한바탕 놀아보세!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/양주별산대놀이'
  },
  
  // 북청사자놀음 (함남 북청)
  bukcheong: {
    name: 'bukcheong',
    koreanName: '북청사자놀음탈',
    origin: '북청사자놀음',
    description: '함경남도 북청 지역의 사자놀음에 사용되는 탈입니다. 정월대보름에 사자탈을 쓰고 집집마다 다니며 악귀를 쫓고 복을 비는 벽사 의식입니다.',
    danceRole: '사자탈을 쓰고 마을의 악귀를 쫓는 벽사 의식 역할',
    quote: '으르렁! 잡귀들아 물러가라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/북청사자놀음'
  }
};

// 탈 이름으로 정보 찾기 (파일명에서 추출된 이름 지원)
export function findMaskInfo(maskName: string): MaskDetailInfo | null {
  // 숫자 제거 및 정규화 (예: 양반1 -> 양반, Yangban2 -> yangban)
  const cleanName = maskName.replace(/\d+/g, '').trim();
  const normalizedName = cleanName.toLowerCase().replace(/[^a-z가-힣]/g, '');
  
  // 정확한 key 매칭 (영문)
  if (MASK_INFO[normalizedName]) {
    return MASK_INFO[normalizedName];
  }
  
  // 한글 이름으로 정확한 매칭
  for (const [key, info] of Object.entries(MASK_INFO)) {
    if (info.koreanName === cleanName || info.koreanName === normalizedName) {
      return info;
    }
  }
  
  // 부분 매칭 (탈 이름이 포함된 경우)
  for (const [key, info] of Object.entries(MASK_INFO)) {
    const koreanNameNormalized = info.koreanName.toLowerCase();
    if (normalizedName.includes(key) || key.includes(normalizedName) ||
        normalizedName.includes(koreanNameNormalized) || koreanNameNormalized.includes(normalizedName)) {
      return info;
    }
  }
  
  return null;
}
